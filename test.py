from utils.pdfdc_utils import extract_pdfdc_segs_metadata
from dataset import load_file, spectrogram_transform, spectrom_feature_extractor, sample_rate
import torch
from video_network import *
import numpy as np
import librosa
import skvideo
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device ='cpu'
device = torch.device(device)
spectrom_feature_extractor = spectrom_feature_extractor.to(device)
EXP_ID = "test_merge"
ALIGN = False
SAVEDIR = Path('saves/') / EXP_ID
SAVEDIR.mkdir(parents=True, exist_ok=True)
def get_reference_set(id_video_list):
    # get a reference set from id_video_list
    # return None if not enough videos
    ref_videos = []
    for video_path, video_label, aug in id_video_list:
        # if true video
        if video_label==1:
            ref_videos.append(video_path)
            # print(video_path)
            
        if len(ref_videos)>=10:
            return ref_videos
    return None

def load_file_pdfdc(filename):
    audio_path = filename[:-3]+'wav'
    if 'crop' not in filename:
        filename = filename[:-4] + '_crop_v2.mp4' 
        if not os.path.exists(filename):
            raise IOError(f"{filename} not cropped")

    video_data = skvideo.io.vread(filename)
    video_data = video_data.transpose((0,3,1,2))
    # evenly sample
    video_size = video_data.shape[0]
    if video_size < 25:
        raise IOError(f" {filename} not enough size")
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
    audio_spectrom  = audio_spectrom
    if not ALIGN:
        video_feature = video_network(video_data_torch)
        audio_feature = audio_network(audio_spectrom)
        return video_feature, audio_feature
    else:
        video_feature, video_align_feature = video_network(video_data_torch)
        audio_feature, audio_align_feature = audio_network(audio_spectrom)
        return video_feature, audio_feature, video_align_feature, audio_align_feature

def get_video_feature(video_path, video_net, audio_net):
    with torch.no_grad():
        seg_features = []
        for seg_file in video_path.iterdir():
            if '.mp4' in str(seg_file) and 'crop' not in str(seg_file):
                if not ALIGN:
                    seg_video_feature, seg_audio_feature = load_feature(seg_file, video_net, audio_net)
                    seg_features.append((seg_video_feature, seg_audio_feature))
                if ALIGN:
                    video_feature, audio_feature, video_align_feature, audio_align_feature = \
                    load_feature(seg_file, video_net, audio_net)
                    seg_features.append((video_feature, audio_feature, video_align_feature, audio_align_feature))

    return seg_features


def calc_feature_distance(ref_video_feature, video_feature, ref_audio_feature, audio_feature):
    # (1, 256)
    return max(torch.norm(ref_video_feature - video_feature, p=2).item(),
     torch.norm(ref_audio_feature - audio_feature, p=2).item())
    # return torch.norm(ref_video_feature - video_feature, p=2).item()
    # return torch.norm(ref_audio_feature - audio_feature, p=2).item()

def calc_align_score(video_align_feature, audio_align_feature):
    return torch.norm(video_align_feature - audio_align_feature, p=2).item()

def get_test_set(ref_set, id_video_list, id_name):
    test_videos_real = []
    test_videos_fake = []
    for video_path, video_label, aug in video_list:
        # if true video
        if video_path in ref_set: #that's ok
            continue
        if video_label==1:
            test_videos_real.append(video_path)
            # print(f'adding real video {video_path}')
        else:
            test_videos_fake.append(video_path)
            # print(f'adding fake video {video_path}')
    print(f"for {id_name}, except for 10 videos for ref, we got {len(test_videos_real)} \
    real videos and {len(test_videos_fake)} fake videos")
    return test_videos_real, test_videos_fake

if __name__=='__main__':
    video_net = ResNetVideo(ALIGN).to(device)
    audio_net = ResNetAudio(ALIGN).to(device)
    # load pth
    #video_net.load_state_dict(torch.load('saves/lr_tune/video_params_11.pth', map_location=device))
    #audio_net.load_state_dict(torch.load('saves/lr_tune/audio_params_11.pth', map_location=device))
    video_net.load_state_dict(torch.load('saves/train_standard/video_params_20.pth', map_location=device))
    audio_net.load_state_dict(torch.load('saves/train_standard/audio_params_20.pth', map_location=device))

    metadata = extract_pdfdc_segs_metadata()
    # map from id to reference set
    print('find refs and tests')
        
    id_ref_dict = {}
    id_test_dict = {}
    for id_name, video_list in metadata.items():
        videos = get_reference_set(video_list)
        if videos:
            id_ref_dict[id_name] = videos
            test_videos_real, test_videos_fake = get_test_set(id_ref_dict[id_name], video_list, id_name)
            if len(test_videos_real)<10:
                print(f'{id_name} not enough real videos')
                continue
            id_test_dict[id_name] = (test_videos_real, test_videos_fake)
        else:
            print(f'{id_name} not enough video for ref')


    # compare
    total_real_distances = []
    total_fake_distances = []
    total_real_scores = []
    total_fake_scores = []
    for id_name, tup in id_test_dict.items():
        print(f'running {id_name}')
        test_videos_real, test_videos_fake = tup
        ref_videos = id_ref_dict[id_name]
        # first calc ref features
        ref_features = []
        for segs_dir in ref_videos:
            seg_features = get_video_feature(segs_dir, video_net, audio_net)
            ref_features.extend(seg_features)

        # then for each video in test set
        real_distances = []
        for test_video in test_videos_real:
            seg_features = get_video_feature(test_video, video_net, audio_net)
            # get minimum distance
            if not ALIGN:
                for video_feature, audio_feature in seg_features:
                    # for each seg_feature, find its min distance
                    distances = []
                    for ref_video_feature, ref_audio_feature in ref_features:    
                        distance = calc_feature_distance(ref_video_feature, video_feature,\
                        ref_audio_feature, audio_feature)
                        distances.append(distance)
                    # also calc score
                    real_distances.append(min(distances))
            else:
                for video_feature, audio_feature, video_align_feature, audio_align_feature in seg_features:
                    # for each seg_feature, find its min distance
                    distances = []
                    for ref_video_feature, ref_audio_feature,_,_ in ref_features:    
                        distance = calc_feature_distance(ref_video_feature, video_feature,\
                        ref_audio_feature, audio_feature)
                        distances.append(distance)
                    # also calc score
                    real_distances.append(min(distances))
        
        # print(real_distances)

        # compare distance
        fake_distances = []
        for test_video in test_videos_fake:
            seg_features = get_video_feature(test_video, video_net, audio_net)
            # get minimum distance
            for video_feature, audio_feature in seg_features:
                distances = []
                for ref_video_feature, ref_audio_feature in ref_features:
                    distance = calc_feature_distance(ref_video_feature, video_feature,\
                     ref_audio_feature, audio_feature)
                    distances.append(distance)

                # also calc score
                fake_distances.append(min(distances))
        
        print(f'for id {id_name}, real distance {np.mean(real_distances)}, fake distance {np.mean(fake_distances)}')
        total_real_distances.extend(real_distances)
        total_fake_distances.extend(fake_distances)
    # AUC
    fpr, tpr, thresholds = roc_curve([1]*len(total_real_distances)+[0]*len(total_fake_distances), \
    total_real_distances + total_fake_distances, pos_label=0)
    auc_score = auc(fpr, tpr)
    print(f'auc: {auc_score}')
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % auc_score,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic at last epoch")
    plt.legend(loc="lower right")
    plt.savefig(str(SAVEDIR / "test_auc.png"))
    plt.clf()
    
