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
from utils.vox_utils import extract_seg_metadata# extract_metadata, 
# https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels

sample_rate = 16000
window_size = sample_rate//1000 * 25 # 25 ms size
hop_size = sample_rate//1000 * 10 # 10 ms hop
mel_bins = 128
# fmin = 50
# fmax = 14000
spectrom_feature_extractor = torch.nn.Sequential(
    tl.Spectrogram(
        n_fft=512, 
        hop_length=hop_size,
        win_length=window_size,
    ), transforms.Resize((224,224)))

def load_file_transform(filename):
    video_data = skvideo.io.vread(filename)
    video_data = video_data.transpose((0,3,1,2))
    # evenly sample
    video_size = video_data.shape[0]
    sample_indexes = np.arange(0,video_size, video_size//25)[:25]
    video_data = video_data[sample_indexes]
    
    audio_path = filename[:-3]+'wav'
    (waveform, _) = librosa.core.load(audio_path, sr = sample_rate, mono=True)
    # NOTE: transform here?
    print(len(waveform))
    S = np.abs(librosa.stft(waveform, n_fft=512, hop_length=hop_size, win_length=window_size))**2
    # S = librosa.feature.melspectrogram(S=D, sr=sample_rate)
    # S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=mel_bins,fmax=fmax)
    # S_dB = librosa.power_to_db(S, ref=np.max)
    return video_data, S

def load_file(filename):
    video_data = skvideo.io.vread(filename)
    video_data = video_data.transpose((0,3,1,2))
    # evenly sample
    video_size = video_data.shape[0]
    sample_indexes = np.arange(0,video_size, video_size//25)[:25]
    video_data = video_data[sample_indexes]
    
    audio_path = filename[:-3]+'wav'
    (waveform, _) = librosa.core.load(audio_path, sr = sample_rate, mono=True)
    return video_data,waveform
    
class LoadFileDataset(torch.utils.data.Dataset):
    def __init__(self, metadata):
        self.metadata = metadata
        self.ids = list(metadata.keys())
        self.pool = ThreadPool(processes=8)
        self.num_id = len(self.ids)
    
    def get_data(self, one_id):
        # 分层抽样
        # sample one id and sample 8 videos
        SAMPLE_NUM = 4
        try:
            # one_id = choice(self.ids)
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
        except KeyboardInterrupt:
            exit(1)
        except SystemExit:
            print('sys exit')
            exit(1)
        except Exception as e:
            print('handling exception')
            print(e)
            return self.get_data()
            

    def __len__(self):
        return self.num_id*16

    def __getitem__(self, idx):
        return self.get_data(self.ids[idx%self.num_id])

def spectrogram_transform(audio_waves: torch.Tensor):
    # input: waveform tensor, batch X length
    # output:resized melspectrogram
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
    ## audio transform

    torch_video_batch = [torch.Tensor(t)[:min_video_length] for t in torch_video_batch]
    torch_audio_batch = [torch.Tensor(t)[:min_audio_length] for t in torch_audio_batch]
    torch_id_batch = torch.LongTensor(torch_id_batch)
    torch_video_batch = pad_sequence(torch_video_batch, batch_first=True)
    torch_audio_batch = pad_sequence(torch_audio_batch, batch_first=True)

    torch_audio_spectrom_batch = spectrogram_transform(torch_audio_batch)
    # list of sequences with size L x *
    # considering memory, encode to
    # print(video_lengths)
    # print(audio_lengths)
    return torch_video_batch, torch_audio_spectrom_batch, torch_id_batch

import cv2
def visualize(img_numpy):
    sample_spec = img_numpy
    print(sample_spec.max())
    print(sample_spec.min())
    sample = sample_spec/sample_spec.max()*255

    cv2.imwrite("sample.jpg", sample.astype(np.uint8))

if __name__=='__main__':
    # NOTE: test audio transform 

    filename = 'YOURPATH/Voxceleb2/test_segs/mp4/id00081/2xYrsnvtUWc/00002/0.mp4'
    video_data, waveform = load_file(filename)
    print(waveform.shape)
    spect = spectrogram_transform(torch.from_numpy(waveform).unsqueeze(0))
    print(spect.size())
    print(spect.max())
    visualize(spect.squeeze().numpy())
    # _, spect = load_file_transform(filename)
    # print(spect.shape)
    # visualize(spect)
    # NOTE: test dataset
    # dataset = LoadFileDataset()
    # loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_padd)
    # print(len(dataset))
    # for video_batch, audio_spectrom_batch, id_batch in loader:
    #     print(video_batch.size())
    #     print(audio_spectrom_batch.size())
    #     print(id_batch.size())
    #     break