import os
import pickle as pk
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad

from litehuman.datasets import MixDataset, MixDatasetCam, PW3D, MixDataset2Cam, MixDataset3Cam, MixDataset2VMCam, H36mSMPLVM, PW3DVM
from litehuman.models import builder
from litehuman.opt import cfg, logger, opt
from litehuman.utils.env import init_dist
from litehuman.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy, vertice_pve
from litehuman.utils.transforms import flip, get_func_heatmap_to_coord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

import time

model = builder.build_sppe(cfg.MODEL)
file_path = 'model_files/litehuman_ckpt/vm_hrnetw48.pth'
pretrain_dict = torch.load(file_path)
# flow2d = model.flow2d
# flow3d = model.flow3d
# pretrain_dict = pretrain_dict['state_dict']
state = {}
for k, v in pretrain_dict.items():
    t = k.split('.')
    if t[0] == 'flow2d' or t[0] == 'flow3d':
        state[k] = v
model_state = model.state_dict()
model_state.update(state)
model.load_state_dict(model_state)
# print(pretrain_dict.keys())