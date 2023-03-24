import librosa
from models import *
from moviepy.editor import *
from pytorch_utils import move_data_to_device

sample_rate = 32000 
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
classes_num = 2



def load_path(video_path):
    (waveform, _) = librosa.core.load(video_path, sr = sample_rate, mono=True)
    print(waveform)
    print(len(waveform))
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, 'cpu')
    batch_output_dict = model(waveform, None)
    return batch_output_dict['embedding']
    
if __name__=='__main__':
    model = Cnn14(sample_rate=sample_rate, window_size=window_size, 
    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    classes_num=classes_num)
    # https://stackoverflow.com/questions/26741116/python-extract-wav-from-video-file
    video_path = 'YOURPATH/Voxceleb/official/test_videos/id00017/01dfn2spqyE/00001.mp4'
    video = VideoFileClip(video_path)
    audio_path = 'test.wav'
    video.audio.write_audiofile(audio_path)
    embed1 = load_path(video_path)
    embed2 = load_path(audio_path)
    print((embed1-embed2).norm())