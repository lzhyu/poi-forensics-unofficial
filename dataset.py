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

BASE_PATH = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/'
sample_rate = 32000 
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
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
    return video_data, waveform
    
class LoadFileDataset(torch.utils.data.Dataset):
    def __init__(self):
        
        self.metadata = extract_metadata()
        self.ids = list(self.metadata.keys())
        self.pool = ThreadPool(processes=8)
        # use hashtable?
    
    def get_data(self):
        # sample one id and sample 8 videos
        one_id = choice(self.ids)
        mp4s = self.metadata[one_id]
        sampled_filenames = random.sample(mp4s, 8)
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

        return (video_datas, audio_datas, [one_id]*8)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.get_data()

# to deal with various length of data, see https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/13
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # input: batch: list of length batch_size
    # each element is a tuple of three lists, each list is a list of 8 arrays
    # output: (batch X length X H X W), (batch X length), batch X 1
    torch_video_batch = []
    torch_audio_batch = []
    torch_id_batch = []
    for data_tuple in batch:
        torch_video_batch.extend(data_tuple[0])
        torch_audio_batch.extend(data_tuple[1])
        torch_id_batch.extend(data_tuple[2])
    
    video_lengths = torch.LongTensor([t.shape[0] for t in torch_video_batch])
    audio_lengths = torch.LongTensor([t.shape[0] for t in torch_audio_batch])
    ## pad
    torch_video_batch = [torch.Tensor(t) for t in torch_video_batch]
    torch_audio_batch = [torch.Tensor(t) for t in torch_audio_batch]
    torch_id_batch = torch.LongTensor(torch_id_batch)
    torch_video_batch = pad_sequence(torch_video_batch, batch_first=True)
    torch_audio_batch = pad_sequence(torch_audio_batch, batch_first=True)
    # list of sequences with size L x *
    print(video_lengths)
    print(audio_lengths)
    return torch_video_batch, torch_audio_batch, torch_id_batch

if __name__=='__main__':
    dataset = LoadFileDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_padd)
    print(len(dataset))
    for video_batch, audio_batch, id_batch in loader:
        print(video_batch.size())
        print(audio_batch.size())
        print(id_batch.size())
        break