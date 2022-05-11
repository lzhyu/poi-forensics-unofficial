
# define loss

# write log
from models import Cnn14
from video_network import *
from dataset import *
from supcon import contrastive_loss
#from brain_test import *
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    # audio_net = Cnn14(sample_rate=sample_rate, window_size=window_size, 
    # hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    # classes_num=classes_num)
    print("audio netting")
    video_net = ResNetVideo().to(device)
    
    print('video netting')
    audio_net = ResNetVideo().to(device)

    # get dataset
    dataset = LoadFileDataset()
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_padd)
    # deal with different wave length and video length
    # load 8 ids per iteration
    # train iter
    for video_batch, audio_spectrogram_batch, id_batch in loader:
        # B X T X C X H X W
        # B X 1 x H X W
        # B
        print(video_batch.size())
        print(audio_spectrogram_batch.size())
        print(id_batch.size())
        video_batch = video_batch.to(device)
        audio_batch = audio_batch.to(device)
        
        video_embeddings = video_net(video_batch)
        print(video_embeddings.size())
        audio_embeddings = audio_net(audio_batch)
        
        print(audio_embeddings.size())
        
        # https://blog.csdn.net/hxxjxw/article/details/121176061
        print(torch.cuda.memory_allocated()/(1024*1024*1024))
        # NOTE: check LOSS
        
        # NOTE：train
        break