import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResNet
from custom.model.loss import MSE
from custom.model.head import RegresionHead
from custom.model.Network import Regression_Network


class network_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # img
    img_size = (160, 160)
    
    # network
    network = Regression_Network(
        backbone = ResNet(
            in_channel=1, 
            block_name="BasicBlock",
            layers=[3, 4, 6, 3]
            ),
        head=RegresionHead(
            num_features_in=512,
            num_classes=1,
        ),
        apply_sync_batchnorm=False
    )


    # loss function
    train_loss_f = MSE()
    valid_loss_f = MSE()
    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            resize(img_size),
            normlize(win_clip=None),
            random_gamma_transform(gamma_range=[0.8, 1.2], prob=0.5),
            random_rotate(theta_range=[-15, 15], prob=0.5)
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            resize(img_size),
            normlize(win_clip=None),
            ])
        )
    
    # train dataloader
    batchsize = 2
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [40,80,120]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 150
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/Resnet50"
    checkpoints_dir = work_dir + '/checkpoints/Resnet50'
    load_from = work_dir + '/checkpoints/Resnet50/none.pth'
