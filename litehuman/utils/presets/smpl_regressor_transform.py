import math
import random

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy,
                          rotmat_to_quat_numpy, flip_twist)
from ..pose_utils import get_intrinsic_metrix

class SMPLRegressorTransform(object):
    """Generation of smpl parameters"""
    def __init__(self, dataset, train):
        self._train = train
        self._joint_pairs_24 = dataset.joint_pairs_24
        
    def __call__(self, label):
        beta = label['beta'].copy()
        theta = label['theta'].copy()
        if 'twist_phi' in label.keys():
            twist_phi = label['twist_phi'].copy()
            twist_weight = label['twist_weight'].copy()
        else:
            twist_phi = np.zeros((23, 2))
            twist_weight = np.zeros((23, 2))
        
        # flip
        if random.random() > 0.5 and self._train:
            theta = flip_thetas(theta, self._joint_pairs_24)
            twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self._joint_pairs_24)

        theta_rot_mat = batch_rodrigues_numpy(theta)
        theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(24 * 4)

        # weight
        target_smpl_weight = torch.ones(1).float()
        theta_24_weights = np.ones((24, 4))
        theta_24_weights = theta_24_weights.reshape(24 * 4)

        output = {
            # 'type': 'smpl_param_data_regressor',
            'target_theta_rotmat': torch.from_numpy(theta_rot_mat).float(),
            'target_theta': torch.from_numpy(theta_quat).float(),
            'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
            'target_beta': torch.from_numpy(beta).float(),
            'target_smpl_weight': target_smpl_weight,
            'target_twist': torch.from_numpy(twist_phi).float(),
            'target_twist_weight': torch.from_numpy(twist_weight).float()
        }
        return output
        