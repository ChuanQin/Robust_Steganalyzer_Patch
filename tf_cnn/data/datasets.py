import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from scipy import io
from glob import glob
from random import randrange

class ImageWithNameDataset(Dataset):

    def __init__(self, img_dir, indices = None, ref_dir = None, transform = None, repeat = 1):
        super(ImageWithNameDataset, self).__init__()

        self.img_dir = img_dir
        self.indices = indices
        self.ref_dir = ref_dir
        full_img_list = sorted(glob(self.img_dir + '/*'))
        if indices is not None:
            self.img_list = [full_img_list[i-1] for i in indices]
        elif ref_dir is not None:
            self.img_list = os.listdir(ref_dir)
        else:
            self.img_list = full_img_list.copy()
        if np.size(np.where(np.asarray(self.img_dir.split('/'))=='cover'))!=0 or \
        any([any(np.asarray(dir_str.split('_'))=='cover') for dir_str in self.img_dir.split('/')]):
            self.label_list = np.zeros(len(self.img_list))
        else:
            self.label_list = np.ones(len(self.img_list))
        self.len = len(self.label_list)
        self.repeat = repeat
        self.transform = transform

    def __getitem__(self, i):
        index = i % self.len
        label = np.array(self.label_list[index])
        if self.ref_dir is None:
            image_path = self.img_list[index]
        else:
            image_path = self.img_dir + '/' + self.img_list[index]
        img = self.transform(Image.open(image_path))
        return img, image_path.split('/')[-1], label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len

class ProbMatDataset(Dataset):

    def __init__(self, img_dir, prob_dir, 
        patch_size = 80, stride = 24, group = 10,
        transform=None):
        self._transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.group = group
        self.images, self.probs, self.labels = self.get_items(img_dir, prob_dir)
        self.len = len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        image = np.expand_dims(image, 2)  # (H, W, C)
        assert image.ndim == 3

        prob = io.loadmat(self.probs[idx])['P']
        assert prob.ndim == 3

        # sort and pack image patches
        prob = np.expand_dims(prob[...,0] + prob[...,1], 2)
        prob_patch = [prob[i:i+self.patch_size,j:j+self.patch_size, :] \
            for i in range(0, image.shape[-3]-self.patch_size, self.stride) \
            for j in range(0, image.shape[-2]-self.patch_size, self.stride)]
        image_patch = [image[i:i+self.patch_size,j:j+self.patch_size, :] \
            for i in range(0, image.shape[-3]-self.patch_size, self.stride) \
            for j in range(0, image.shape[-2]-self.patch_size, self.stride)]
        prob_sums = [p.sum() for p in prob_patch]
        prob_idx = np.argsort(prob_sums)
        # dividing into 10 groups
        grp_idx = list(self.divide(prob_idx.tolist(), self.group))
        image, prob = image_patch[grp_idx[0][randrange(len(grp_idx[0]))]], prob_patch[grp_idx[0][randrange(len(grp_idx[0]))]]
        for i in range(1, self.group):
            image = np.concatenate((image, image_patch[grp_idx[i][randrange(len(grp_idx[i]))]]), -1)
            prob = np.concatenate((prob, prob_patch[grp_idx[i][randrange(len(grp_idx[i]))]]), -1)

        sample = {
            'image': image,
            'prob': prob,
            # 'data': data,
            'label': self.labels[idx]
        }

        if self._transform:
            sample = self._transform(sample)
        return sample, self.images[idx].split('/')[-1]

    @staticmethod
    def divide(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    @staticmethod
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def get_items(img_dir, prob_dir):
        images, prob_mtx, labels = [], [], []

        img_names = sorted(os.listdir(img_dir))
        prob_names = sorted(os.listdir(prob_dir))
        for i,p in zip(img_names,prob_names):
            img_path = osp.join(img_dir, i)
            images.append(img_path)
            if 'cover' in img_path.split('/'):
                labels.append(0)
            prob_path = osp.join(prob_dir,p)
            prob_mtx.append(prob_path)

        return images, prob_mtx, labels