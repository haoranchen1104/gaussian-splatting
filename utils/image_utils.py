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

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import torch
import numpy as np


SAM_BIT_LEN = 32

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_sam_mask(file_path, resolution):
    sam_mask = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    sam_mask = cv2.resize(sam_mask, resolution, interpolation=cv2.INTER_NEAREST)
    sam_mask = sam_mask.view(np.uint32)
    sam_mask = sam_mask.astype(np.int64)
    sam_mask = torch.from_numpy(sam_mask)
    return sam_mask

def get_masks_from_sam_mask(sam_mask):
    masks = []
    areas = []
    for i in range(SAM_BIT_LEN):
        mask = torch.bitwise_and(1 << i, sam_mask).bool()
        masks.append(mask)
        areas.append(mask.sum())
    return masks, areas

