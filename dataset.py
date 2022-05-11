# load dataset
# don't need to split train and test
# but should test the loading time
# use generator to load dataset, see https://blog.csdn.net/cjs8348797/article/details/115708811
import torch
from pathlib import Path
from random import choice
import random
import librosa
from torch.utils.data import DataLoader
import skvideo.io
from torch.nn.utils.rnn import pad_sequence
import threading
from multiprocessing.pool import ThreadPool
import torchlibrosa as tl
from torchvision import transforms
import numpy as np
# https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
BASE_PATH = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/'
sample_rate = 22050#16000 
window_size = 2048#400
hop_size = 512#160
mel_bins = 128
fmin = 50
fmax = 14000
spectrom_feature_extractor = torch.nn.Sequential(
    tl.Spectrogram(
        hop_length=hop_size,
        win_length=window_size,
    ), tl.LogmelFilterBank(
        sr=sample_rate,
        n_mels=mel_bins,
        is_log=False, # Default is true
    ),transforms.Resize((224,224)))

def extract_metadata():
    # maintain a dict = {id: {video:[clips]}}
    metadata = {}
    path = Path(BASE_PATH)
    for child in path.iterdir():
        if child.is_dir():
            id_number = int(child.name[2:])
            metadata[id_number] = []
            #load videos
            for video_dirs in child.iterdir():
                for seg in video_dirs.glob("*.mp4"):
                    metadata[id_number].append(str(seg.absolute()))
        else:
            print('why here?')
            print(child.name)
    return metadata

def load_file(filename):
    video_data = skvideo.io.vread(filename)
    video_data = video_data.transpose((0,3,1,2))
    
    audio_path = filename[:-3]+'wav'
    (waveform, _) = librosa.core.load(audio_path, sr = sample_rate, mono=True)
    # transform here
    # S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=mel_bins,
    #                                 fmax=fmax)
    # S_dB = librosa.power_to_db(S, ref=np.max)
    return video_data,waveform
    
class LoadFileDataset(torch.utils.data.Dataset):
    def __init__(self):
        
        self.metadata = extract_metadata()
        self.ids = list(self.metadata.keys())
        self.pool = ThreadPool(processes=8)
        # use hashtable?
    
    def get_data(self):
        # sample one id and sample 8 videos
        SAMPLE_NUM = 4
        one_id = choice(self.ids)
        mp4s = self.metadata[one_id]
        sampled_filenames = random.sample(mp4s, SAMPLE_NUM)
        video_datas = []
        audio_datas = []
        async_res_list = []
        # multiprocessing copied from https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
        
        for filename in sampled_filenames:# multithread
            # print(filename)
            async_result = self.pool.apply_async(load_file, (filename, ))
            async_res_list.append(async_result)
            # video_data, waveform = load_file(filename)

        for async_result in async_res_list:
            video_data, waveform = async_result.get()
            video_datas.append(video_data)
            audio_datas.append(waveform)

        return (video_datas, audio_datas, [one_id]*SAMPLE_NUM)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.get_data()

def spectrogram_transform(audio_waves: torch.Tensor):
    # input: waveform tensor, batch X length
    # output:resized melspectrogram
    print(audio_waves.max())
    print(audio_waves.min())
    resized_spectrom = spectrom_feature_extractor(audio_waves)
    return resized_spectrom
    # (batch_size, 1, time_steps, mel_bins)

# to deal with various length of data, see https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/13
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # input: batch: list of length batch_size
    # each element is a tuple of three lists, each list is a list of 8 arrays
    # output: (batch X length X H X W), (batch X H X W), batch X 1
    torch_video_batch = []
    torch_audio_batch = []
    torch_id_batch = []
    for data_tuple in batch:
        torch_video_batch.extend(data_tuple[0])
        torch_audio_batch.extend(data_tuple[1])
        torch_id_batch.extend(data_tuple[2])
    
    video_lengths = torch.LongTensor([t.shape[0] for t in torch_video_batch])
    min_video_length = min([t.shape[0] for t in torch_video_batch])
    audio_lengths = torch.LongTensor([t.shape[0] for t in torch_audio_batch])
    min_audio_length = min([t.shape[0] for t in torch_audio_batch])
    ## pad
    ## TODO: audio transform

    torch_video_batch = [torch.Tensor(t)[:min_video_length] for t in torch_video_batch]
    torch_audio_batch = [torch.Tensor(t)[:min_audio_length] for t in torch_audio_batch]
    torch_id_batch = torch.LongTensor(torch_id_batch)
    torch_video_batch = pad_sequence(torch_video_batch, batch_first=True)
    torch_audio_batch = pad_sequence(torch_audio_batch, batch_first=True)

    torch_audio_spectrom_batch = spectrogram_transform(torch_audio_batch)
    # list of sequences with size L x *
    # considering memory, encode to
    print(video_lengths)
    print(audio_lengths)
    return torch_video_batch, torch_audio_spectrom_batch, torch_id_batch

import cv2
def visualize(img_numpy):
    sample_spec = img_numpy
    sample = sample_spec/sample_spec.max()*255
    print(sample.max())
    print(sample.min())
    cv2.imwrite("sample.jpg", sample.astype(np.uint8))

if __name__=='__main__':
    # filename = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/id00017/01dfn2spqyE/00001.mp4'
    # vd,ad = load_file(filename)
    # print(ad.max())
    # print(ad.min())
    # print(ad.shape)
    # import cv2
    # cv2.imwrite("sample.jpg", ad.astype(np.uint8))
    dataset = LoadFileDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_padd)
    print(len(dataset))
    for video_batch, audio_spectrom_batch, id_batch in loader:
        print(video_batch.size())
        print(audio_spectrom_batch.size())
        print(id_batch.size())
        break