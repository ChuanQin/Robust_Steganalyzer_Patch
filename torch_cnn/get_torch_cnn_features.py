import numpy as np
import pickle
import torch
import torchvision
import argparse
import torch.nn as nn
import models
import data
from scipy import io
import os
from utils import get_CNN_feature

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cover_dir', type=str, required=True,)
    parser.add_argument('--stego_dir', type=str, required=True,)
    parser.add_argument('--adv_dir', type=str, required=True,)
    parser.add_argument('--cover_feature_path', type=str, required=True,)
    parser.add_argument('--stego_feature_path', type=str, required=True,)
    parser.add_argument('--adv_feature_path', type=str, required=True,)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

def set_dataloader(args):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
    ])
    cover_data = data.ImageWithNameDataset(
                    img_dir = args.cover_dir, 
                    transform = transform)
    stego_data = data.ImageWithNameDataset(
                    img_dir = args.stego_dir, 
                    transform = transform)
    adv_data = data.ImageWithNameDataset(
                    img_dir = args.adv_dir, 
                    ref_dir = args.cover_dir, 
                    transform = transform)
    cover_loader = torch.utils.data.DataLoader(
                    cover_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    stego_loader = torch.utils.data.DataLoader(
                    stego_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    adv_loader = torch.utils.data.DataLoader(
                    adv_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)

    return cover_loader, stego_loader, adv_loader

def set_model(args):
    net = nn.DataParallel(models.KeNet())
    ckpt_path = args.ckpt_dir + '/model_best.pth.tar'
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()

    return net

args = parse_args()
net = set_model(args)
cover_loader, stego_loader, adv_loader = set_dataloader(args)
cover_features, names = get_CNN_feature(net, cover_loader)
stego_features, names = get_CNN_feature(net, stego_loader)
adv_features, names = get_CNN_feature(net, adv_loader)

if not os.path.exists('/'.join(args.cover_feature_path.split('/')[:-1])):
    os.makedirs('/'.join(args.cover_feature_path.split('/')[:-1]))
io.savemat(args.cover_feature_path, {'F': cover_features, 'names': names})
io.savemat(args.stego_feature_path, {'F': stego_features, 'names': names})
io.savemat(args.adv_feature_path, {'F': adv_features, 'names': names})