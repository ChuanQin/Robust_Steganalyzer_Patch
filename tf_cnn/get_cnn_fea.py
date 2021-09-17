import torch
import tensorflow as tf
import torchvision
import os
import argparse
from scipy import io
import data
from ext_fea_lib import ext_cnn_fea

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, required=True,)
    parser.add_argument('--img_dir', type=str, required=True,)
    parser.add_argument('--prob_dir', type=str, required=True,)
    parser.add_argument('--feature_path', type=str, required=True,)
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=80)
    parser.add_argument('--group', type=int, default=10)
    parser.add_argument('--stride', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

def set_dataloader(args):
    # num_image = len(os.listdir(args.cover_dir))
    # random_images = np.arange(0, num_image)
    # np.random.seed(3)
    # np.random.shuffle(random_images)
    transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            data.ToTensor()
    ])
    img_data = data.ProbMatDataset(
                    img_dir = args.img_dir,
                    prob_dir = args.prob_dir, 
                    transform = transform)
    img_loader = torch.utils.data.DataLoader(
                    img_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)

    return img_loader

args = parse_args()
img_loader = set_dataloader(args)
# get cover features
features, names = ext_cnn_fea(args.model_type, img_loader, args.load_path, args.batch_size)

# save features
if not os.path.exists('/'.join(args.feature_path.split('/')[:-1])):
    os.makedirs('/'.join(args.feature_path.split('/')[:-1]))
io.savemat(args.feature_path, {'F':features, 'names': names})