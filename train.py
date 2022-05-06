
# define loss

# write log
from audio_net import *
from video_network import *
from dataset import *
from supcon import contrastive_loss
from brain_test import *
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    # audio_net = Cnn14(sample_rate=sample_rate, window_size=window_size, 
    # hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    # classes_num=classes_num)
    audio_net = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",\
    freeze_params=False)
    video_net = ResNetVideo()

    # get dataset
    dataset = LoadFileDataset()
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn_padd)
    # deal with different wave length and video length
    # load 8 ids per iteration
    # train iter
    for video_batch, audio_batch, id_batch in loader:
        # B X T X C X H X W
        # B X T 
        # B
        print(video_batch.size())
        print(audio_batch.size())
        print(id_batch.size())
        # audio_embeddings = audio_net(audio_batch)['embedding']
        audio_embeddings = audio_net
        video_embeddings = video_net.encode_batch(signal)
        # video_embeddings = video_net(video_batch)
        # FIXME: why killed here
        print(audio_embeddings.size())
        print(video_embeddings.size())
        break
        batch_output_dict = model(waveform, None)