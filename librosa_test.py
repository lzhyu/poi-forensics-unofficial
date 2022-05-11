import numpy as np
import librosa
sample_rate = 22050
audio_path = '00001.wav'
(waveform, _) = librosa.core.load(audio_path, sr = sample_rate, mono=True)
# transform here
S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128,
                                fmax=8000)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.savefig("test.png")