import torchvision.models as models
from torch import nn
import torch
import skvideo.io  

# resize input to 224
class ResNetVideo(nn.Module):
    # B*T*C*H*W
    def __init__(self, align):
        
        super(ResNetVideo, self).__init__()
        self.align = align
        self.resnet18 = models.resnet18(pretrained = False)
        self.linear = nn.Linear(1000, 256)
        if self.align:
            self.align_linear = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
                )
        
    def forward(self, input):
        # input B,T,C,H,W
        batch_size, length, channels, H, W = input.size()
        reshaped_input = input.reshape(batch_size*length, channels, H, W)
        features = self.resnet18(reshaped_input.float())
        if self.align:
            align_features = self.align_linear(features)
            align_features = align_features.reshape(batch_size, length, -1)
        features = self.linear(features)
        features = features.reshape(batch_size, length, -1)
        if self.align:
            return features.mean(dim=1), align_features.mean(dim=1)
        return features.mean(dim=1)

class ResNetAudio(nn.Module):
    # B*T*C*H*W
    def __init__(self, align=False):
        
        super(ResNetAudio, self).__init__()
        self.align = align
        self.resnet18 = models.resnet18(pretrained = False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linear = nn.Linear(1000, 256)
        if self.align:
            self.align_linear = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
                )
        
    def forward(self, input):
        # input B,T,C,H,W
        batch_size, length, channels, H, W = input.size()
        reshaped_input = input.reshape(batch_size*length, channels, H, W)
        features = self.resnet18(reshaped_input.float())
        if self.align:
            align_features = self.align_linear(features)
            align_features = align_features.reshape(batch_size, length, -1)
        features = self.linear(features)
        features = features.reshape(batch_size, length, -1)
        if self.align:
            return features.mean(dim=1), align_features.mean(dim=1)
        return features.mean(dim=1)

class ResNet50Video(nn.Module):
    # B*T*C*H*W
    def __init__(self):
        
        super(ResNet50Video, self).__init__()
        self.resnet50 = models.resnet50(pretrained = False)
        self.linear = nn.Linear(1000, 256)
        
    def forward(self, input):
        # input B,T,C,H,W
        batch_size, length, channels, H, W = input.size()
        reshaped_input = input.reshape(batch_size*length, channels, H, W)
        features = self.resnet50(reshaped_input.float())
        features = self.linear(features)
        features = features.reshape(batch_size, length, -1)
        return features.mean(dim=1)

if __name__=='__main__':
    test = ResNetAudio()
    # video_path = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/id00017/01dfn2spqyE/00001.mp4'
    # videodata = skvideo.io.vread(video_path)
    # videodata = videodata.transpose((0,3,1,2))
    # videodata = videodata[None, :]
    # model = ResNetVideo()
    # print('before cuda')
    # model = model.to('cuda')
    # print('after cuda')
    # res = model(torch.from_numpy(videodata).to('cuda'))
    # print(res.size())
    # print(videodata.shape)
