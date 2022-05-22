from utils.pdfdc_utils import extract_pdfdc_segs_metadata
from dataset import load_file, spectrogram_transform, spectrom_feature_extractor, sample_rate
import torch
from video_network import *
import numpy as np
import librosa
import skvideo

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device ='cpu'
device = torch.device(device)
spectrom_feature_extractor = spectrom_feature_extractor.to(device)
def get_reference_set(id_video_list):
    # get a reference set from id_video_list
    # return None if not enough videos
    ref_videos = []
    for video_path, video_label, aug in id_video_list:
        # if true video
        if video_label==1:
            ref_videos.append(video_path)
            print(video_path)
            
        if len(ref_videos)>=10:
            return ref_videos
    return None

def load_file_pdfdc(filename):
    audio_path = filename[:-3]+'wav'
    if 'crop' not in filename:
        filename = filename[:-4] + '_crop.mp4' 

    video_data = skvideo.io.vread(filename)
    video_data = video_data.transpose((0,3,1,2))
    # evenly sample
    video_size = video_data.shape[0]
    sample_indexes = np.arange(0,video_size, video_size//25)[:25]
    video_data = video_data[sample_indexes]
    
    
    (waveform, _) = librosa.core.load(audio_path, sr = sample_rate, mono=True)
    return video_data,waveform

def load_feature(file_path, video_network, audio_network):
    video_data, waveform = load_file_pdfdc(str(file_path))
    # video_data = (? X length X H X W)
    waveform_torch = torch.from_numpy(waveform)[None,...].to(device)
    video_data_torch = torch.from_numpy(video_data)[None,...].to(device)
    
    audio_spectrom = spectrogram_transform(waveform_torch)[None,...]
    audio_spectrom  = audio_spectrom.repeat((1,1,3,1,1))
    video_feature = video_network(video_data_torch)
    audio_feature = audio_network(audio_spectrom)
    return video_feature, audio_feature

def get_video_feature(video_path, video_net, audio_net):
    with torch.no_grad():
        seg_video_features = []
        seg_audio_features = []
        for seg_file in video_path.iterdir():
            if '.mp4' in str(seg_file) and 'crop' not in str(seg_file):
                seg_video_feature, seg_audio_feature = load_feature(seg_file, video_net, audio_net)
                seg_video_features.append(seg_video_feature.unsqueeze(0))
                seg_audio_features.append(seg_audio_feature.unsqueeze(0))
        video_feature = torch.cat(seg_video_features).mean(dim=0)
        audio_feature = torch.cat(seg_audio_features).mean(dim=0)

    return video_feature, audio_feature


def calc_feature_distance(feature_1, feature_2):
    # (1, 256)
    return torch.norm(feature_1 - feature_2, p=2)

def get_test_set(ref_set, id_video_list, id_name):
    test_videos_real = []
    test_videos_fake = []
    for video_path, video_label, aug in video_list:
        # if true video
        if video_path in ref_set: #that's ok
            continue
        if video_label==1:
            test_videos_real.append(video_path)
            print(f'adding real video {video_path}')
        else:
            test_videos_fake.append(video_path)
            print(f'adding fake video {video_path}')
    print(f"for {id_name}, except for 10 videos for ref, we got {len(test_videos_real)} \
    real videos and {len(test_videos_fake)} fake videos")
    return test_videos_real, test_videos_fake

if __name__=='__main__':
    print('begin loading model')
    video_net = ResNetVideo().to(device)
    audio_net = ResNetVideo().to(device)
    # load pth
    video_net.load_state_dict(torch.load('saves/video_params_293.pth', map_location=device))
    audio_net.load_state_dict(torch.load('saves/audio_params_293.pth', map_location=device))

    metadata = extract_pdfdc_segs_metadata()
    # map from id to reference set
    print('find refs')
    id_ref_dict = {}
    for id_name, video_list in metadata.items():
        videos = get_reference_set(video_list)
        if videos:
            id_ref_dict[id_name] = videos
            break
        else:
            print(f'{id_name} not enough video for ref')
        

    # for id in ids, find test set
    print('finding test set')
    id_test_dict = {}
    for id_name, video_list in metadata.items():
        if id_name not in id_ref_dict:
            continue
        test_videos_real, test_videos_fake = get_test_set(id_ref_dict[id_name], video_list, id_name)
        id_test_dict[id_name] = (test_videos_real, test_videos_fake)

    # compare
    print('compare distances')
    for id_name, tup in id_test_dict.items():
        test_videos_real, test_videos_fake = tup
        ref_videos = id_ref_dict[id_name]
        # first calc ref features
        print('calculating refs')
        ref_features = []
        for segs_dir in ref_videos:
            video_feature, audio_feature = get_video_feature(segs_dir, video_net, audio_net)
            ref_features.append((video_feature, audio_feature))

        # then for each video in test set
        real_distances = []
        for test_video in test_videos_real:
            video_feature, audio_feature = get_video_feature(test_video, video_net, audio_net)
            # get minimum distance
            min_distance = 10000
            distances = []
            for ref_video_feature, ref_audio_feature in ref_features:
                distance = calc_feature_distance(ref_video_feature, video_feature).item()
                distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
            # real_distances.append(min_distance)
            real_distances.append(np.mean(distances))
        print(test_videos_real)
        print(real_distances)

        # compare distance
        fake_distances = []
        for test_video in test_videos_fake:
            video_feature, audio_feature = get_video_feature(test_video, video_net, audio_net)
            # get minimum distance
            min_distance = 10000
            distances = []
            for ref_video_feature, ref_audio_feature in ref_features:
                distance = calc_feature_distance(ref_video_feature, video_feature).item()
                distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
            # fake_distances.append(min_distance)
            fake_distances.append(np.mean(distances))
            print(f'fake video distance: {min_distance}')
        print(fake_distances)
        print(f'for id {id_name}, real distance {np.mean(real_distances)}, fake distance {np.mean(fake_distances)}')