# write log
# 现在用验证集训，大概一晚上到收敛
from models import Cnn14
from video_network import *
from dataset import *
from supcon import contrastive_loss
import torch.nn.functional as F
import torch
from utils.vox_utils import extract_metadata, extract_seg_metadata

def compute_correct_rate(id_batch:torch.Tensor, nearest_indices:torch.Tensor):
    correct_nearest = 0
    for k in range(nearest_indices.size()[0]):
        if id_batch[k].item()==id_batch[nearest_indices[k][1].item()].item():
            correct_nearest +=1
    return correct_nearest/(4*4)

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device ='cpu'
    # load model
    # audio_net = Cnn14(sample_rate=sample_rate, window_size=window_size, 
    # hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    # classes_num=classes_num)
    print('begin')
    video_net = ResNetVideo().to(device)
    audio_net = ResNetVideo().to(device)
    # get dataset
    # split ids to train and valid
    metadata = extract_seg_metadata()
    metadata_train = dict(list(metadata.items())[:int(len(metadata)*0.8)])
    metadata_valid = dict(list(metadata.items())[int(len(metadata)*0.8):])
    
    train_dataset = LoadFileDataset(metadata_train)
    valid_dataset = LoadFileDataset(metadata_valid)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn_padd)
    valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn_padd)
    optimizer = torch.optim.Adam(list(video_net.parameters())+list(audio_net.parameters()), \
    lr=1e-4, weight_decay=0.01)
    # deal with different wave length and video length
    # load 8 ids per iteration
    # train iter
    # 12 epochs with 2304 batches per epoch,
    train_batchs = 0
    print('begin training')
    video_loss_list = []
    audio_loss_list = []
    total_loss_list = []
    video_correct_rate_list = []
    audio_correct_rate_list = []
    video_loss_list_valid = []
    audio_loss_list_valid = []
    video_correct_rate_list_valid = []
    audio_correct_rate_list_valid = []
    for epoch in range(50):
        print(epoch)
        # train
        for video_batch, audio_spectrogram_batch, id_batch in train_loader:
            train_batchs+=1
            # B X T X C X H X W
            # B X 1 x H X W
            # B
            video_batch = video_batch.to(device)
            audio_batch = audio_spectrogram_batch.to(device).repeat((1,3,1,1)).unsqueeze(1)
            id_batch = id_batch.unsqueeze(1).to(device)
            video_embeddings = video_net(video_batch)
            video_embeddings = F.normalize(video_embeddings,p=2,dim=1)*3
            # batch, feature_dim
            audio_embeddings = audio_net(audio_batch)
            audio_embeddings = F.normalize(audio_embeddings,p=2,dim=1)*3
            
            # https://blog.csdn.net/hxxjxw/article/details/121176061
            # print(torch.cuda.memory_allocated()/(1024*1024*1024))
            # 8G gpu memory
            optimizer.zero_grad()
            video_loss, nearest_indices_video = contrastive_loss(video_embeddings, id_batch, device=device)
            # nearest_indices_video compare to id_batch
            video_correct_rate_list.append(compute_correct_rate(id_batch, nearest_indices_video))

            audio_loss, nearest_indices_audio  = contrastive_loss(audio_embeddings, id_batch, device=device)
            audio_correct_rate_list.append(compute_correct_rate(id_batch, nearest_indices_audio))
            total_loss = video_loss + audio_loss
            video_loss_list.append(video_loss.item())
            audio_loss_list.append(audio_loss.item())
            total_loss_list.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

        if epoch%10==3:
            print(epoch)
            print(train_batchs)
            print(f'total loss {np.mean(total_loss_list)}')
            print(f'video loss {np.mean(video_loss_list)}')
            print(f'audio loss {np.mean(audio_loss_list)}')
            print(f'video correct rate {np.mean(video_correct_rate_list)}')
            print(f'audio correct rate {np.mean(audio_correct_rate_list)}')
            torch.save(video_net.state_dict(), f'saves/video_params_{epoch}.pth')
            torch.save(audio_net.state_dict(), f'saves/audio_params_{epoch}.pth')
            video_loss_list = []
            audio_loss_list = []
            total_loss_list = []
            video_correct_rate_list = []
            audio_correct_rate_list = []
        
        torch.cuda.empty_cache()
        # valid

        for video_batch, audio_spectrogram_batch, id_batch in valid_loader:
            # B X T X C X H X W
            # B X 1 x H X W
            # B
            with torch.no_grad():
                video_batch = video_batch.to(device)
                audio_batch = audio_spectrogram_batch.to(device).repeat((1,3,1,1)).unsqueeze(1)
                id_batch = id_batch.unsqueeze(1).to(device)
                video_embeddings = video_net(video_batch)
                video_embeddings = F.normalize(video_embeddings,p=2,dim=1)*3
                audio_embeddings = audio_net(audio_batch)
                audio_embeddings = F.normalize(audio_embeddings,p=2,dim=1)*3
                
                optimizer.zero_grad()
                video_loss, nearest_indices_video = contrastive_loss(video_embeddings, id_batch, device=device)
                # nearest_indices_video compare to id_batch
                video_correct_rate_list_valid.append(compute_correct_rate(id_batch, nearest_indices_video))

                audio_loss, nearest_indices_audio  = contrastive_loss(audio_embeddings, id_batch, device=device)
                audio_correct_rate_list_valid.append(compute_correct_rate(id_batch, nearest_indices_audio))

                video_loss_list_valid.append(video_loss.item())
                audio_loss_list_valid.append(audio_loss.item())

        if epoch%10==3:
            print(f'video loss valid {np.mean(video_loss_list_valid)}')
            print(f'audio loss valid {np.mean(audio_loss_list_valid)}')
            print(f'video correct rate valid {np.mean(video_correct_rate_list_valid)}')
            print(f'audio correct rate valid {np.mean(audio_correct_rate_list_valid)}')
            video_loss_list_valid = []
            audio_loss_list_valid = []
            video_correct_rate_list_valid = []
            audio_correct_rate_list_valid = []