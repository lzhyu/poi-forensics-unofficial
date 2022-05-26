# 现在用验证集训，大概一晚上到收敛
# from models import Cnn14
from video_network import *
from dataset import *
from supcon import contrastive_loss
import torch.nn.functional as F
import torch
from utils.vox_utils import extract_seg_metadata
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

EXP_ID = "train_standard_lr"
SAVEDIR = Path('saves/') / EXP_ID
SAVEDIR.mkdir(parents=True, exist_ok=True)
NORMALIZE = False
ALIGN = False
LR = 1e-4
print(f'exp_id: {EXP_ID}, align:{ALIGN}, learning rate:{LR}')
# distances
def compute_correct_rate(id_batch:torch.Tensor, nearest_indices:torch.Tensor):
    correct_nearest = 0
    for k in range(nearest_indices.size()[0]):
        if id_batch[k].item()==id_batch[nearest_indices[k][1].item()].item():
            correct_nearest +=1
    return correct_nearest/(4*4)

def compute_sample_auc(id_batch:torch.Tensor, similarity_zero:torch.Tensor):
    id_same = (id_batch==id_batch[0]).to(torch.int).cpu().numpy()
    similarity_zero = similarity_zero.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(id_same, similarity_zero, pos_label=1)
    auc_score = auc(fpr, tpr)

    return auc_score

def plot_two_curve(train_list, valid_list, title, ylabel):
    plt.clf()
    plt.plot(train_list, alpha=0.5, color='#a68f00', label='train')
    plt.plot(valid_list, alpha=0.5, label='valid')
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel("epoch")
    plt.title(title)
    plt.savefig(str(SAVEDIR / title)+'.png')
    
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device ='cpu'
    # load model

    print('begin')
    video_net = ResNetVideo(ALIGN).to(device)
    audio_net = ResNetAudio(ALIGN).to(device)
    # get dataset
    # split ids to train and valid
    metadata = extract_seg_metadata()
    metadata_train = dict(list(metadata.items())[:int(len(metadata)*0.8)])
    metadata_valid = dict(list(metadata.items())[int(len(metadata)*0.8):])
    
    train_dataset = LoadFileDataset(metadata_train)
    valid_dataset = LoadFileDataset(metadata_valid)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn_padd, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=collate_fn_padd, shuffle=True)
    optimizer = torch.optim.Adam(list(video_net.parameters())+list(audio_net.parameters()), \
    lr=LR, weight_decay=0.01)
    # deal with different wave length and video length
    # load 8 ids per iteration
    # train iter
    # 12 epochs with 2304 batches per epoch,
    train_batchs = 0
    print('begin training')
    video_loss_plot = []
    audio_loss_plot = []
    total_loss_plot = []
    align_loss_plot = []
    video_metric_plot = []
    audio_metric_plot = []

    video_loss_plot_valid = []
    audio_loss_plot_valid = []
    align_loss_plot_valid = []
    video_metric_plot_valid = []
    audio_metric_plot_valid = []

    for epoch in range(41):
        print(f'epoch:{epoch}')
        video_loss_list = []
        audio_loss_list = []
        total_loss_list = []
        align_loss_list = []
        video_metric_list = []
        audio_metric_list = []

        video_loss_list_valid = []
        audio_loss_list_valid = []
        align_loss_list_valid = []
        video_metric_list_valid = []
        audio_metric_list_valid = []
        # train
        for video_batch, audio_spectrogram_batch, id_batch in train_loader:
            # if num_id <=2, continue
            if torch.unique(id_batch).size()[0]<=2:
                continue

            train_batchs+=1
            # B X T X C X H X W
            # B X 1 x H X W
            # B
            video_batch = video_batch.to(device)
            audio_batch = audio_spectrogram_batch.to(device).repeat((1,1,1,1)).unsqueeze(1)
            id_batch = id_batch.unsqueeze(1).to(device)

            if ALIGN:
                video_embeddings, video_align_features = video_net(video_batch)
            else:
                video_embeddings = video_net(video_batch)

            if NORMALIZE:
                video_embeddings = F.normalize(video_embeddings,p=2,dim=1)*3
            # batch, feature_dim

            optimizer.zero_grad()
            # video_loss, nearest_indices_video = contrastive_loss(video_embeddings, id_batch, device=device)
            # video_metric_list.append(compute_correct_rate(id_batch, nearest_indices_video))

            video_loss, similarity_zero_video = contrastive_loss(video_embeddings, id_batch, device=device)
            video_metric_list.append(compute_sample_auc(id_batch, similarity_zero_video))
            video_loss_list.append(video_loss.item())
            if not ALIGN:
                video_loss.backward()
                optimizer.step()

            # https://blog.csdn.net/hxxjxw/article/details/121176061
            # print(torch.cuda.memory_allocated()/(1024*1024*1024))
            # 8G gpu memory
            
            optimizer.zero_grad()
            if ALIGN:
                audio_embeddings, audio_align_features = audio_net(audio_batch)
            else:
                audio_embeddings = audio_net(audio_batch)
            if NORMALIZE:
                audio_embeddings = F.normalize(audio_embeddings,p=2,dim=1)*3
            audio_loss, similarity_zero_audio = contrastive_loss(audio_embeddings, id_batch, device=device)
            audio_metric_list.append(compute_sample_auc(id_batch, similarity_zero_audio))

            # audio_metric_list.append(compute_correct_rate(id_batch, nearest_indices_audio))
            
            audio_loss_list.append(audio_loss.item())
            if not ALIGN:
                audio_loss.backward()
                optimizer.step()
            
            #align loss
            if ALIGN:
                optimizer.zero_grad()
                align_loss = torch.norm(video_align_features - audio_align_features, p=2)
                # add temperature?
                (align_loss + video_loss + audio_loss).backward()
                optimizer.step()
                align_loss_list.append(align_loss.item())

            total_loss_list.append(video_loss_list[-1] + audio_loss_list[-1])

        video_loss_plot.append(np.mean(video_loss_list))
        audio_loss_plot.append(np.mean(audio_loss_list))
        video_metric_plot.append(np.mean(video_metric_list))
        audio_metric_plot.append(np.mean(audio_metric_list))
        total_loss_plot.append(np.mean(total_loss_list))
        if ALIGN:
            align_loss_plot.append(np.mean(align_loss_list))
        if epoch%5==0:
            print(train_batchs)
            print(f'total loss {total_loss_plot[-1]}')
            print(f'video loss {video_loss_plot[-1]}')
            print(f'audio loss {audio_loss_plot[-1]}')
            print(f'video metric {video_metric_plot[-1]}')
            print(f'audio metric {audio_metric_plot[-1]}')
            if ALIGN:
                print(f'align loss {align_loss_plot[-1]}')
            torch.save(video_net.state_dict(), str(SAVEDIR / f'video_params_{epoch}.pth'))
            torch.save(audio_net.state_dict(), str(SAVEDIR / f'audio_params_{epoch}.pth'))
        
        
        # valid

        for video_batch, audio_spectrogram_batch, id_batch in valid_loader:
            if torch.unique(id_batch).size()[0]<=2:
                continue
            # B X T X C X H X W
            # B X 1 x H X W
            # B
            with torch.no_grad():
                video_batch = video_batch.to(device)
                audio_batch = audio_spectrogram_batch.to(device).repeat((1,1,1,1)).unsqueeze(1)
                id_batch = id_batch.unsqueeze(1).to(device)
                if ALIGN:
                    video_embeddings, video_align_features = video_net(video_batch)
                else:
                    video_embeddings = video_net(video_batch)
                if NORMALIZE:
                    video_embeddings = F.normalize(video_embeddings,p=2,dim=1)*3

                if ALIGN:
                    audio_embeddings, audio_align_features = audio_net(audio_batch)
                else:
                    audio_embeddings = audio_net(audio_batch)
                if NORMALIZE:
                    audio_embeddings = F.normalize(audio_embeddings,p=2,dim=1)*3
                
                video_loss, similarity_zero_video = contrastive_loss(video_embeddings, id_batch, device=device)
                video_metric_list_valid.append(compute_sample_auc(id_batch, similarity_zero_video))
                # video_metric_list_valid.append(compute_correct_rate(id_batch, nearest_indices_video))

                audio_loss, similarity_zero_audio = contrastive_loss(audio_embeddings, id_batch, device=device)
                audio_metric_list_valid.append(compute_sample_auc(id_batch, similarity_zero_audio))
                # audio_metric_list_valid.append(compute_correct_rate(id_batch, nearest_indices_audio))

                video_loss_list_valid.append(video_loss.item())
                audio_loss_list_valid.append(audio_loss.item())

                if ALIGN:
                    align_loss = torch.norm(video_align_features - audio_align_features, p=2)
                    align_loss_list_valid.append(align_loss.item())

        video_loss_plot_valid.append(np.mean(video_loss_list_valid))
        audio_loss_plot_valid.append(np.mean(audio_loss_list_valid))
        video_metric_plot_valid.append(np.mean(video_metric_list_valid))
        audio_metric_plot_valid.append(np.mean(audio_metric_list_valid))
        if ALIGN:
            align_loss_plot_valid.append(np.mean(align_loss_list_valid))

        if epoch%5==0:
            print(f'video loss valid {video_loss_plot_valid[-1]}')
            print(f'audio loss valid {audio_loss_plot_valid[-1]}')
            print(f'video metric valid {video_metric_plot_valid[-1]}')
            print(f'audio metric valid {audio_metric_plot_valid[-1]}')
            if ALIGN:
                print(f'align loss valid {align_loss_plot[-1]}')

    # plot results
    plot_two_curve(video_loss_plot, video_loss_plot_valid, 'Video Loss Curve', 'loss')
    plot_two_curve(audio_loss_plot, audio_loss_plot_valid, 'Audio Loss Curve', 'loss')
    plot_two_curve(video_metric_plot, video_metric_plot_valid, 'Video Sampled AUC Curve', 'AUC')
    plot_two_curve(audio_metric_plot, audio_metric_plot_valid, 'Audio Sampled AUC Curve', 'AUC')
    if ALIGN:
        plot_two_curve(align_loss_plot, align_loss_plot_valid, 'Align Loss Curve', 'loss')
    

