import torchvision.models as models
from torch import nn
import torch
import skvideo.io  

# resize input to 224
class ResNetVideo(nn.Module):
    # B*T*C*H*W
    def __init__(self):
        
        super(ResNetVideo, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        
    def forward(self, input):
        # input B,T,C,H,W
        batch_size, length, channels, H, W = input.size()
        reshaped_input = input.reshape(batch_size*length, channels, H, W)
        features = self.resnet18(reshaped_input.float())
        features = features.reshape(batch_size, length, -1)
        return features.mean(dim=1)

if __name__=='__main__':
    video_path = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/id00017/01dfn2spqyE/00001.mp4'
    videodata = skvideo.io.vread(video_path)
    videodata = videodata.transpose((0,3,1,2))
    videodata = videodata[None, :]
    model = ResNetVideo()
    res = model(torch.from_numpy(videodata))
    print(res.size())
    print(videodata.shape)
