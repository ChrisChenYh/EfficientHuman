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

from .layers.graph_layers import GraphLinear, GraphResBlock
from .layers.smpl_beta_regressor import SMPLBetaRegressor

import time


def xyz2uvd(xyz_jts, trans, intrinsic_param, joint_root, depth_factor, return_relative=True):
    xyz_jts = xyz_jts * depth_factor.unsqueeze(-1)
    if return_relative:
        xyz_jts = xyz_jts + joint_root.unsqueeze(1)

    # (B, K, 1)
    abs_z = xyz_jts[:, :, 2]
    xyz_jts = xyz_jts / abs_z.unsqueeze(-1)

    dz = abs_z - joint_root[:, 2].unsqueeze(-1)

    intrinsic_param_inv = intrinsic_param.inverse()
    xyz_jts = xyz_jts.unsqueeze(-1)

    # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
    cam_2d_homo = torch.matmul(intrinsic_param_inv.unsqueeze(1), xyz_jts)
    # uv_jts = cam_2d_homo[: , :, :2, :]

    # print(uv_jts.shape)
    # print(trans.shape)
    # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
    uv_homo_jts = torch.matmul(trans.unsqueeze(1), cam_2d_homo)
    uv_homo_jts = uv_homo_jts.squeeze(dim=3)
    uv_homo_jts = uv_homo_jts[:, :, :2]

    # print(uv_homo_jts.shape)
    # print(dz.shape)

    uvd_jts = torch.cat((uv_homo_jts, dz.unsqueeze(-1)), dim=2)
    uvd_jts[:, :, 2] = uvd_jts[:, :, 2] / depth_factor

    uvd_jts[:, :, 0] = uvd_jts[:, :, 0] / 64 / 4 - 0.5
    uvd_jts[:, :, 1] = uvd_jts[:, :, 1] / 64 / 4 - 0.5

    return uvd_jts

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))

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
class RLEGraph(nn.Module):
    def __init__(self, A='model_files/vertices54_adj.npy', norm_layer=nn.BatchNorm2d, **kwargs):
        super(RLEGraph, self).__init__()
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

        # Posenet Head
        self.avg_pool_0 = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(self.feature_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 3)
        # self.fc_layers = [self.fc_coord, self.fc_sigma]
        
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

        #### Graph CNN Layers
        self.gc_feature_channels = 512
        adj = np.load(A)
        adj = torch.from_numpy(adj)
        self.A = adj.to_sparse().to(torch.device('cuda'))
        graphcnn_layers = self._make_graphcnn_layer(A=self.A, input_channels=self.feature_channel, num_layers=5, num_channels=self.gc_feature_channels)
        self.gc_encoder = nn.Sequential(*graphcnn_layers)
        self.gc_veritces_coord = nn.Sequential( GraphResBlock(self.gc_feature_channels, 64, self.A),
                                                GraphResBlock(64, 32, self.A),
                                                nn.GroupNorm(32 // 8, 32),
                                                nn.ReLU(inplace=True),
                                                GraphLinear(32, 3))
        self.beta_regressor = SMPLBetaRegressor()

    def _initialize(self):
        for m in self.fc_coord:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_sigma:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)            

    def frozen_weights(self, frozen_layers):
        # frozen the weights of:
        # preact, 
        for layer in frozen_layers:
            layer = 'self.' + layer
            layer = eval(layer)
            for i in layer.parameters():
                i.requires_grad = False

    def _make_graphcnn_layer(self, A, input_channels=512, num_layers=5, num_channels=512):
        layers = [GraphLinear(3 + input_channels, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        return layers

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

    def forward(self, x, labels=None, mesh_sampler=None, device=torch.device('cuda'), flip_test=False, **kwargs):

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
        if labels is not None:
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
            # print(nf_loss.shape)
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
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # shape time
        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

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
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

            pred_shape = (pred_shape + flip_pred_shape) / 2

            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            flip_pred_camera[:, 1] = -flip_pred_camera[:, 1]
            pred_camera = (pred_camera + flip_pred_camera) / 2

        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
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
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)

        # between_time_end = time.time()
        # print('between inference time: {}'.format(between_time_end - between_time_start))              

        # smpl inference start time
        smpl_start = time.time()
        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        # smpl_end = time.time()
        # print('smpl inference time: {}'.format(smpl_end - smpl_start))

        ##### add shape calibrate part
        xg = feat
        ## use graph cnn to estimate the coarsen human mesh vertices
        # get the template mesh with estimated pose (output.rot_mats) and template shape
        # output.rotmats -> theta
        tempalte_beta = torch.zeros((batch_size, 10), dtype=self.smpl_dtype).cuda(device)
        
        # [32, 96]
        # print(output.rot_mats.shape)
        # get the template mesh template (use predicted theta)
        ref_output = self.smpl(
            pose_axis_angle=output.rot_mats,
            betas=tempalte_beta,
            global_orient=None,
            transl=None,
            return_verts=True
        )
        ref_vertices = ref_output.vertices.float()
        # template mesh simplification
        ref_vertices_sub = mesh_sampler.downsample(ref_vertices)
        ref_vertices_sub2 = mesh_sampler.downsample(ref_vertices_sub, n1=1, n2=2)
        ref_vertices_sub3 = mesh_sampler.downsample(ref_vertices_sub2, n1=2, n2=3)
        ref_vertices_sub4 = mesh_sampler.downsample(ref_vertices_sub3, n1=3, n2=4)
        
        # Have not do the coord transform, just pred the vertices coord in smpl space
        # [B, 54, 3] -> [B, 3, 54]
        ref_vertices_sub4 = ref_vertices_sub4.permute(0, 2, 1) / 2.2
        # [B, 512, 1] -> [b, 512, 54]
        xg = xg.view(batch_size, self.feature_channel, 1).expand(-1, -1, ref_vertices_sub4.shape[-1])
        # [B, 512, 54] -> [B, 512 + 3, 54]
        xg = torch.cat([ref_vertices_sub4, xg], dim=1)
        # print(xg.shape)
        # GCNN encoder
        xg = self.gc_encoder(xg)
        # Get the calibrated sub vertices
        pred_vertices_sub4 = self.gc_veritces_coord(xg)
        # Get the delta vertices coord
        # print(pred_vertices_sub4.shape)
        # print(ref_vertices_sub4.shape)
        delta_vertives_sub4 = pred_vertices_sub4 - ref_vertices_sub4
        # Use MLP to solve out the beta with delta vertices as input
        cal_beta = self.beta_regressor(delta_vertives_sub4)
        # Get the calibrated vertices
        cal_output = self.smpl(
            pose_axis_angle=output.rot_mats,
            betas=cal_beta,
            global_orient=None,
            transl=None,
            return_verts=True
        )
        cal_vertices = cal_output.vertices.float()
        
        cal_xyz_jts_24_struct = cal_output.joints.float() / self.depth_factor
        cal_xyz_jts_17 = cal_output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        cal_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        cal_xyz_jts_24_struct = cal_xyz_jts_24_struct.reshape(batch_size, 72)
        cal_xyz_jts_17_flat = cal_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = camera_root - cal_output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        ## 
        # pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        # pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        #  -0.5 ~ 0.5
        # pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        # pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        # pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        # pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        # pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        # transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        # [B, 3, 54] -> [B, 54, 3]
        pred_vertices_sub4 = pred_vertices_sub4.permute(0, 2, 1)

        output = edict(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=cal_xyz_jts_24,
            pred_xyz_jts_24_struct=cal_xyz_jts_24_struct,
            pred_xyz_jts_17=cal_xyz_jts_17_flat,
            pred_vertices=cal_vertices,
            pred_vertices_sub4=pred_vertices_sub4,
            maxvals=scores.float(),
            sigma=sigma,
            nf_loss=nf_loss,
            log_phi=log_phi,
            log_sigma=log_sigma,
            log_phi_2d=log_phi_2d,
            log_phi_3d=log_phi_3d,
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
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
