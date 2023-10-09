import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
from easydict import EasyDict as edict
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.smpl.SMPL import SMPL_layer
from .layers.real_nvp import RealNVP


import time

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))

def flip_coord(preds, joint_pairs, width_dim, shift=True, flatten=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores, pred_sigma = preds
    if flatten:
        assert pred_jts.dim() == 2 and pred_scores.dim() == 3 and pred_sigma.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3 and pred_scores.dim() == 3 and pred_sigma.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]

    # flip
    if shift:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0] - 1 / (width_dim * 4)
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]
        pred_sigma[:, idx] = pred_sigma[:, inv_idx]

    # pred_jts = pred_jts.reshape(num_batches, num_joints * 3)
    return pred_jts, pred_scores, pred_sigma

def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError

def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))

def nets3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3), nn.Tanh())

def nett3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3))

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y

@SPPE.register_module
class Simple3DPoseBaseSMPLRLE(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLRLE, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        backbone = ResNet

        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            ''' Load pretrained model '''
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        self.avg_pool_0 = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(self.feature_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 3)
        self.fc_layers = [self.fc_coord, self.fc_sigma]
        
        # flow
        self.share_flow = True
        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))        
        prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))
        self.flow2d = RealNVP(nets, nett, masks, prior)
        self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_smpl = 0

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9, 0, 0])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float()) 
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.focal_length = kwargs['FOCAL_LENGTH']
        self.bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.depth_factor = float(self.bbox_3d_shape[2]) * 1e-3
        self.input_size = 256.0

    def _initialize(self):
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]

        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]

        return heatmaps

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def forward(self, x, labels=None, flip_test=False, is_train=True, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        feat = self.avg_pool_0(x0).reshape(batch_size, -1)
        out_coord = self.fc_coord(feat).reshape(batch_size, self.num_joints, 3)
        assert out_coord.shape[2] == 3

        out_sigma = self.fc_sigma(feat).reshape(batch_size, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(batch_size, self.num_joints, 3)
        sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma
        scores = torch.mean(scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_feat = self.avg_pool_0(flip_x0).reshape(batch_size, -1)
            flip_out_coord = self.fc_coord(flip_feat).reshape(batch_size, self.num_joints, 3)
            assert flip_out_coord.shape[2] == 3
            flip_out_sigma = self.fc_sigma(flip_feat).reshape(batch_size, self.num_joints, -1)
            flip_pred_jts = flip_out_coord.reshape(batch_size, self.num_joints, 3)
            flip_sigma = flip_out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
            flip_scores = 1 - flip_sigma
            flip_scores = torch.mean(flip_scores, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_scores, flip_sigma]
            flip_pred_jts, flip_scores, flip_sigma = flip_coord( flip_preds, 
                                                                 joint_pairs=self.joint_pairs_29,
                                                                 width_dim=self.width_dim,
                                                                 shift=True,
                                                                 flatten=False)
            
            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            scores = (scores + flip_scores) / 2
            sigma = (sigma + flip_sigma) / 2
            
        # compute the rle loss
        if labels is not None and is_train:
            gt_uvd = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_weight = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

            assert pred_jts.shape == sigma.shape, (pred_jts.shape, sigma.shape)
            # * gt_uvd_weight
            bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, self.num_joints, 1)
            # print(log_phi.shape)
            # print(sigma.shape)
            # sigma [B, 29, 3] log_phi [B, 29, 1]
            log_sigma = torch.log(sigma)
            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None
            # 
            log_phi = None
            log_sigma = None
            log_phi_2d = None
            log_phi_3d = None

        pred_uvd_jts_29 = pred_jts

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        # init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # shape time
        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        
        pred_phi = self.decphi(xc)
        # pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)

            flip_xc = self.fc1(flip_x0)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)

            flip_delta_shape = self.decshape(flip_xc)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_xc)

            pred_shape = (pred_shape + flip_pred_shape) / 2

            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2
        # print(kwargs['depth_factor'])
        pred_xyz_jts_29 = self.uvd_to_cam( pred_uvd_jts_29, 
                                           trans_inv=kwargs['trans_inv'], 
                                           intrinsic_param=kwargs['intrinsic_param'],
                                           joint_root=kwargs['joint_root'],
                                           depth_factor=kwargs['depth_factor'])
        assert torch.sum(torch.isnan(pred_xyz_jts_29)) == 0, ('pred_xyz_jts_29', pred_xyz_jts_29)

        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, self.root_idx_smpl, :].unsqueeze(1)

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        # smpl_end = time.time()
        # print('smpl inference time: {}'.format(smpl_end - smpl_start))        

        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor    # 2.2
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor  # 2.2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        output = edict(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=scores.float(),
            sigma=sigma,
            nf_loss=nf_loss,
            log_phi=log_phi,
            log_sigma=log_sigma,
            log_phi_2d=log_phi_2d,
            log_phi_3d=log_phi_3d,
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
