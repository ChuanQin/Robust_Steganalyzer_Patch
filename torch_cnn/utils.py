import torch
import models
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
from scipy import io

def preprocess_data(images):
    # images of shape: NxCxHxW
    if images.dim() == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
    h, w = images.shape[-2:]
    ch, cw, h0, w0 = h, w, 0, 0
    cw = cw & ~1
    inputs = [
        images[..., h0:h0 + ch, w0:w0 + cw // 2],
        images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
    ]
    # if args.cuda:
    inputs = [x.cuda() for x in inputs]
    return inputs

def get_CNN_feature(
    model,
    dataloader):
    features = np.ndarray(shape = [dataloader.dataset.len, 513])
    names = []
    acc = 0.
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs = preprocess_data(data[0]*255.)
            if isinstance(model, models.KeNet) or isinstance(model.module, models.KeNet):
                output_batch, fea_left, fea_right = model(*inputs)
                feats = torch.stack([fea_left, fea_right], dim=0)
                feats_mean = feats.mean(dim=0)
                feats_var = feats.var(dim=0)
                feats_min, _ = feats.min(dim=0)
                feats_max, _ = feats.max(dim=0)
                euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6,
                                                 keepdim=True)
                final_feat = torch.cat(
                    [euclidean_distance, feats_mean, feats_var, feats_min, feats_max], dim=-1
                    )
            else:
                _, feature = model(inputs)
            features[idx*dataloader.batch_size:(idx+1)*dataloader.batch_size,:]\
                = final_feat.cpu().numpy()
            acc += models.accuracy(output_batch, data[2].cuda()).item()
            names.extend(list(data[1]))
    print('Accuracy on this dataset: {:.4f}'.format(acc/len(dataloader)))
    return features, names