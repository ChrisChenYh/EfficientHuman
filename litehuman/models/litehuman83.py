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
from .smpl_param_regressor import SMPLParamRegressor, get_regressor

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
class Litehuman83(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Litehuman83, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        # Add num_verts 54
        self.num_verts = kwargs['NUM_VERTS']
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

        # Regressor pretrained Model
        self.regressor = get_regressor(cfg=kwargs['REGRESSOR'])

        # Posenet Head (jts + verts)
        self.avg_pool_0 = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(self.feature_channel, (self.num_joints+self.num_verts) * 3)
        self.fc_sigma = nn.Linear(self.feature_channel, (self.num_joints+self.num_verts) * 3)
        self.fc_layers = [self.fc_coord, self.fc_sigma]
        
        # flow
        self.share_flow = True
        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2), validate_args=False)
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))        
        prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3), validate_args=False)
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))
        self.flow2d = RealNVP(nets, nett, masks, prior)
        self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)
        self.flow54 = RealNVP(nets3d, nett3d, masks3d, prior3d)

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
        # self.decshape = nn.Linear(1024, 10)
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

    def forward(self, x, ref_verts=None, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        feat = self.avg_pool_0(x0).reshape(batch_size, -1)
        out_coord = self.fc_coord(feat).reshape(batch_size, (self.num_joints+self.num_verts), 3)
        assert out_coord.shape[2] == 3

        out_sigma = self.fc_sigma(feat).reshape(batch_size, (self.num_joints+self.num_verts), -1)

        # (B, N, 3)
        pred_jts = out_coord[:, :self.num_joints, :]
        pred_verts = out_coord[:, self.num_joints:, :]
        jts_sigma = out_sigma[:, :self.num_joints, :]
        jts_sigma = jts_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        verts_sigma = out_sigma[:, self.num_joints:, :]
        verts_sigma = verts_sigma.reshape(batch_size, self.num_verts, -1).sigmoid() + 1e-9
        # sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        jts_scores = 1 - jts_sigma
        verts_scores = 1 - verts_sigma
        jts_scores = torch.mean(jts_scores, dim=2, keepdim=True)
        verts_scores = torch.mean(verts_scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_feat = self.avg_pool_0(flip_x0).reshape(batch_size, -1)
            flip_out_coord = self.fc_coord(flip_feat).reshape(batch_size, (self.num_joints+self.num_verts), 3)
            assert flip_out_coord.shape[2] == 3
            flip_out_sigma = self.fc_sigma(flip_feat).reshape(batch_size, (self.num_joints+self.num_verts), -1)
            flip_pred_jts = flip_out_coord[:, :self.num_joints, :]
            flip_pred_verts = flip_out_coord[:, self.num_joints:, :]
            flip_jts_sigma = flip_out_sigma[:, :self.num_joints, :]
            flip_jts_sigma = flip_jts_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9

            flip_jts_scores = 1 - flip_jts_sigma
            flip_jts_scores = torch.mean(flip_jts_scores, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_jts_scores, flip_jts_sigma]
            flip_pred_jts, flip_jts_scores, flip_jts_sigma = flip_coord( flip_preds, 
                                                                         joint_pairs=self.joint_pairs_29,
                                                                         width_dim=self.width_dim,
                                                                         shift=True,
                                                                         flatten=False)
            
            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            jts_scores = (jts_scores + flip_jts_scores) / 2
            jts_sigma = (jts_sigma + flip_jts_sigma) / 2

        if labels is not None:
            gt_uvd = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_weight = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

            gt_uvd54 = kwargs['gt_uvd54'].reshape(pred_verts.shape)
            gt_uvd54_weight = labels['target_weight_vertices_sub4'].reshape(pred_verts.shape)
            gt_verts_mask = gt_uvd54_weight[:, :, 2].reshape(-1)
            
            assert pred_jts.shape == jts_sigma.shape, (pred_jts.shape, jts_sigma.shape)
            # * gt_uvd_weight
            bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / jts_sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]

            bar_mu_all = (pred_verts - gt_uvd54) * gt_uvd54_weight / verts_sigma
            bar_mu_all = bar_mu_all.reshape(-1, 3)
            bar_mu_verts = bar_mu_all[gt_verts_mask > 0]            
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, self.num_joints, 1)
            # (B, V, 3)
            log_phi_verts = self.flow54.log_prob(bar_mu_verts)
            log_phi_verts_all = torch.zeros_like(bar_mu_all[:, 0])
            log_phi_verts_all[gt_verts_mask > 0] = log_phi_verts
            log_phi_verts_all = log_phi_verts_all.reshape(batch_size, self.num_verts, 1)            

            # print(log_phi.shape)
            # print(sigma.shape)
            # sigma [B, 29, 3] log_phi [B, 29, 1]
            nf_jts_loss = torch.log(jts_sigma) - log_phi
            nf_verts_loss = torch.log(verts_sigma) - log_phi_verts_all
            # print(nf_loss.shape)
        else:
            nf_jts_loss = None
            nf_verts_loss = None
            # 
            log_phi = None
            log_sigma = None
            log_phi_2d = None
            log_phi_3d = None
            log_phi_verts_all = None

        pred_uvd_jts_29 = pred_jts
        pred_uvd_verts_54 = pred_verts

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        # init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        # phi and cam forward pass
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        # Flip cam and phi
        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)

            flip_xc = self.fc1(flip_x0)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)

            flip_pred_phi = self.decphi(flip_xc)
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            flip_pred_camera[:, 1] = -flip_pred_camera[:, 1]
            pred_camera = (pred_camera + flip_pred_camera) / 2        
        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        # UVD -> XYZ
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        pred_xyz_verts_54 = torch.zeros_like(pred_uvd_verts_54)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = (bboxes[:, 2] - bboxes[:, 0])
            h = (bboxes[:, 3] - bboxes[:, 1])

            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h

            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)

            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_verts_54[:, :, 2:] = pred_uvd_verts_54[:, :, 2:].clone()
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m
            pred_xy_verts_54_meter = ((pred_uvd_verts_54[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_verts_54[:, :, 2:] * self.depth_factor + camDepth)

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_verts_54[:, :, :2] = pred_xy_verts_54_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_verts_54[:, :, 2:] = pred_uvd_verts_54[:, :, 2:].clone()

            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m
            pred_xyz_verts_54_meter = (pred_uvd_verts_54[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_verts_54[:, :, 2:] * self.depth_factor + camDepth) - camTrans

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_verts_54[:, :, :2] = pred_xyz_verts_54_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_verts_54 = pred_xyz_verts_54 - pred_xyz_jts_29[:, [0]]

        pred_xyz_verts_54 = pred_xyz_verts_54 * 2.2

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_xyz_verts_54_flat = pred_xyz_verts_54.reshape(batch_size, -1)

        # Regressor forward pass
        reg_inps = torch.cat([pred_xyz_verts_54, ref_verts], dim=-1)
        pred_params = self.regressor(reg_inps)
        pred_beta = pred_params['pred_shape']

        # Flip beta
        if flip_test:
            # Flip parameters
            flip_reg_inps = torch.cat([flip_pred_verts, ref_verts], dim=-1)
            flip_pred_params = self.regressor(flip_reg_inps)
            flip_pred_beta = flip_pred_params['pred_shape']
            pred_beta = (pred_beta + flip_pred_beta) / 2

        # HybrIK
        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, # unit: meter
            betas=pred_beta.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        # smpl_end = time.time()
        # print('smpl inference time: {}'.format(smpl_end - smpl_start))        

        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        output = edict(
            pred_phi=pred_phi,
            pred_shape=pred_beta,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            pred_uvd54_verts=pred_uvd_verts_54,
            jts_maxvals=jts_scores.float(),
            verts_maxvals=verts_scores.float(),
            jts_sigma=jts_sigma,
            verts_sigma=verts_sigma,
            nf_jts_loss=nf_jts_loss,
            nf_verts_loss=nf_verts_loss,
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
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
