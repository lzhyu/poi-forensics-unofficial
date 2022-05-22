import torchaudio
from speechbrain.pretrained import EncoderClassifier
from torch.optim import Adam
if __name__=='__main__':
    # copied from https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",\
    freeze_params=False)
    signal, fs =torchaudio.load('test.wav')
    classifier.train()
    embeddings = classifier.encode_batch(signal)
    opt = Adam(classifier.parameters())
    loss = embeddings.sum()
    loss.backward()
    print(loss)
    print(type(embeddings))