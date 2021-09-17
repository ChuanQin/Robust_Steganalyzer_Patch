# -*- coding: utf-8 -*-

"""
2018.11
Qin Chuan
"""

from __future__ import print_function

import os
import sys
import time
import numpy as np
from config import *
##### Network selection #####
from SRNet import SRNet
# from SCA_SRNet_Spatial import *
from YeNet import YeNet
# from XuNet import *
#############################
from generator import gen_flip_and_rot, gen_valid
from utils import train, AdamaxOptimizer
from functools import partial

# create randome indices for dividing training/testing/validation sets
random_images = np.arange(0, args.num_image)
np.random.seed(args.seed)
np.random.shuffle(random_images)

im_train = random_images[0 : args.num_train]
im_valid = random_images[args.num_train : args.num_train + args.num_valid]
im_test = random_images[args.num_train + args.num_valid : args.num_image]
# im_train = random_images[10000 : 10000 + args.num_train]
# im_valid = random_images[10000 + args.num_train : 10000 + args.num_train + args.num_valid]
# im_test = random_images[10000 + args.num_train + args.num_valid : args.num_image]
###################### training ######################
# create image reading generator
##### SRNet and YeNet generator #####
train_gen = partial(gen_flip_and_rot,
                    args.cover_dir,
                    args.stego_dir,
                    im_train)
# sca gen
# train_gen = partial(gen_flip_and_rot_sca,
#                     args.cover_dir,
#                     args.stego_dir,
#                     im_train)
######### XuNet generator #########
# train_gen = partial(gen_train,
#                     args.cover_dir,
#                     args.stego_dir,
#                     im_train)
###################################
valid_gen = partial(gen_valid,
                    args.cover_dir,
                    args.stego_dir,
                    im_valid)
                    # im_train)
# valid_gen = partial(gen_valid, 
#                     args.cover_dir,
#                     args.stego_dir,
#                     im_test)
# sca gen
# valid_gen = partial(gen_valid_sca, 
#                     args.cover_dir,
#                     args.stego_dir,
#                     im_valid)
#                     # im_train)

valid_ds_size = len(im_valid) * 2
if(valid_ds_size % args.valid_batch_size != 0):
    raise ValueError("change batch size for validation")

print(args)
train(
      ##### Network selection #####
      model_class = YeNet, 
      # model_class = SRNet,
      # model_class = SCA_SRNet, 
      # model_class = XuNet, 
      #############################
      train_gen = train_gen, 
      valid_gen = valid_gen, 
      train_batch_size = args.train_batch_size, 
      valid_batch_size = args.valid_batch_size, 
      valid_ds_size = valid_ds_size, 
      ##### SRNet: Adamax optimizer #####
      # optimizer = AdamaxOptimizer, 
      ##### YeNet: Adadelta optimizer #####
      optimizer = tf.train.AdadeltaOptimizer, 
      ##### XuNet: Momentum optimizer #####
      # optimizer = tf.train.MomentumOptimizer, 
      #####################################
      boundaries = args.boundaries, 
      values = args.values, 
      train_interval = args.train_interval, 
      valid_interval = args.valid_interval, 
      max_iter = args.max_iter, 
      save_interval = args.valid_interval, 
      log_path = args.log_path, 
      num_runner_threads = 1, 
      ##### load existing network or not #####
      load_path = args.load_path)
      # load_path = None)

# save_prob(model_class = SRNet, 
#           batch_size = args.valid_batch_size, 
#           img_dir = args.img_dir, 
#           prob_dir = args.prob_dir, 
#           prob_mtx = args.prob_mtx, 
#           load_path = args.load_path)

# sca_test(model_class = SCA_SRNet, 
#          img_dir = args.adv_stego_dir,
#          batch_size = args.valid_batch_size, 
#          ds_size = len(random_images), 
#          load_path = args.load_path, 
#          indices = random_images, 
#          payload = args.payload)

###################### testing ######################
# test_ds_size = len(im_test) * 2
# # test_gen = partial(gen_valid, 
# #                    args.cover_dir,
# #                    args.stego_dir,
# #                    im_test)
# test_gen = partial(gen_test, 
#                 args.stego_dir, 
#                 im_test)
# start_time = time.time()
# test_dataset(
#              ##### Network selection #####
#              model_class = SRNet, 
#              # model_class = YeNet, 
#              # model_class = XuNet, 
#              #############################
#              gen = test_gen, 
#              batch_size = args.valid_batch_size, 
#              ds_size = test_ds_size, 
#              load_path = args.load_path)
# end_time = time.time()
# time_len = end_time - start_time
# print('{} images per second'.format(test_ds_size/time_len))
# single_img_gen = partial(gen_single_img,
#                          args.img_dir)
# test_single_img(
#                 ##### Network selection #####
#                 # model_class = SRNet, 
#                 model_class = YeNet, 
#                 gen = single_img_gen, 
#                 load_path = args.load_path
#                 )
# feature_HPF(model_class = SRNet, 
#             load_path = args.load_path, 
#             cover_dir = args.cover_dir, 
#             stego_dir = args.stego_dir, 
#             adv_dir = args.adv_stego_dir, 
#             display_dir = args.display_dir)
# calc_mmd(model_class = SRNet,
#          load_path = args.load_path,
#          cover_dir = args.cover_dir,
#          stego_dir = args.stego_dir,
#          adv_stego_dir = args.adv_stego_dir,
#          batch_size = args.valid_batch_size,
#          ds_size = len(random_images))

# import torchvision.transforms as transforms
# from py_dataset import Pair_Steg_Dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     ])
# test_dataset = Pair_Steg_Dataset(
#                     root = args.root, 
#                     method = args.method, 
#                     indices = im_test, 
#                     payload = args.payload, 
#                     transform = transform)
