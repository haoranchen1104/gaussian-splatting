#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def contrastive_loss(rendered_feature, masks, num_anchor=32, temperature=0.1):
    loss = 0
    num_valid_mask = 0
    for idx in range(len(masks)):
        mask = masks[idx].to(rendered_feature.device)
        if torch.sum(mask) == 0:
            continue
        num_valid_mask += 1
        
        mask_idxs = torch.argwhere(mask)
        anchor_idxs = torch.randint(low=0, high=mask_idxs.shape[0], size=(num_anchor, ))
        anchor_features = rendered_feature[:, mask_idxs[anchor_idxs, 0], mask_idxs[anchor_idxs, 1]]
        anchor_features = anchor_features.permute(1, 0)
        
        pos_features = rendered_feature[:, mask_idxs[:, 0], mask_idxs[:, 1]]
        pos_features = torch.mean(pos_features, dim=1).detach()
        
        neg_mask_idxs = torch.argwhere(torch.logical_not(mask))
        neg_idxs = torch.randint(low=0, high=neg_mask_idxs.shape[0], size=(num_anchor*15, ))
        neg_features = rendered_feature[:, neg_mask_idxs[neg_idxs, 0], neg_mask_idxs[neg_idxs, 1]]
        neg_features = neg_features.permute(1, 0)
        neg_features = neg_features.reshape(num_anchor, 15, -1)
        neg_features = neg_features.detach()

        logits_pos = F.cosine_similarity(anchor_features, pos_features[None, :], dim=-1)
        logits_neg = F.cosine_similarity(anchor_features[:, None, :], neg_features, dim=-1)
        logits = torch.cat((logits_pos[:, None], logits_neg), dim=1)

        labels = torch.zeros(anchor_features.shape[0], dtype=torch.int64).to(rendered_feature.device)
        loss += F.cross_entropy(logits/temperature, labels)
    
    loss /= num_valid_mask
    return loss

