import torch
import tensorflow as tf
import numpy as np
from models import SRNet, YeNet
import os

def ext_cnn_fea(model_type,
    data_loader,
    load_path,
    batch_size,
    patch_size = 80,
    stat_fea = True):
    """
    extract CNN features
    target directory or dataset is defined by the data_loader
    """
    tf.reset_default_graph()
    # prepare model
    if model_type == 'SRNet':
        model = SRNet(is_training = False, img_size = patch_size)
    elif model_type == 'YeNet':
        model = YeNet(is_training = False, img_size = patch_size)
    else:
        raise ValueError('unknown model')
    output = model._build_model()

    # load from a checkpoint
    saver = tf.train.Saver(max_to_keep=10000)
    sess = tf.InteractiveSession()
    if load_path is not None:
        if os.path.isdir(load_path):
            saver.restore(sess, tf.train.latest_checkpoint(load_path))
        else:
            saver.restore(sess, load_path)

    # define how to get features
    feature_batch = model._get_features()
    features = np.empty(shape=(data_loader.dataset.len, data_loader.dataset.group, feature_batch.shape[1]))
    names = []

    # # prepare an iteration
    # data_iter = iter(data_loader)
    # batch_size = data_iter.batch_sampler.batch_size

    for idx, data in enumerate(data_loader):
        img_batch = data[0]['image'].numpy().transpose((0,2,3,1))*255.
        name_batch = data[1]
        for i in range(img_batch.shape[-1]):
            features[idx*img_batch.shape[0]:min(data_loader.dataset.len,(idx+1)*img_batch.shape[0]),i,:] = \
                np.squeeze(sess.run(feature_batch, feed_dict = {model.x_input: np.expand_dims(img_batch[:,:,:,i],-1)}))
        names.extend(name_batch)
    sess.close()
    return features, names