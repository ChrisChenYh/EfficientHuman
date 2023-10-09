import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
from easydict import EasyDict as edict
from torch.nn import functional as F

from .builder import SPPE
from .layers.Resnet import ResNet
from .layers.hrnet import HRNet
from .layers.smpl.SMPL import SMPL_layer
from .layers.real_nvp import RealNVP

from .layers.rtmcc_block import ScaleNorm, RTMCCBlock
from .layers.hrnet import get_hrnet
from .layers.MGFENet import MGFENet
import time
import yaml

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

class RTMCCHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 input_size, 
                 in_featuremap_size, 
                 simcc_split_ratio=2.0, 
                 final_layer_kernel_size=1,
                 gau_cfg=dict(
                    hidden_dim=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    acf_fn='ReLU',
                    use_rel_bias=False,
                    pos_enc=False
                 )):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        # Difine SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2
        )
        # self.mlp = nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)
        )
        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc']
        )

        self.cls_u_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)
        self.cls_v_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)
        self.cls_d_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)
        self.cls_u_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)
        self.cls_v_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)
        self.cls_d_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=False)

        self.cls_head = [self.cls_u_coord, self.cls_u_sigma, self.cls_u_coord, self.cls_u_sigma, self.cls_d_coord, self.cls_d_sigma]

    def forward(self, feats):
        batch_size = feats.shape[0]

        feats = self.final_layer(feats)     # -> B, K, H, W
        
        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)             # -> B, K, hidden

        feats = self.gau(feats)             # -> B, K, hidden

        pred_u_coord = self.cls_u_coord(feats)
        # .reshape(batch_size, 29, 1)
        pred_u_sigma = self.cls_u_sigma(feats)
        pred_v_coord = self.cls_v_coord(feats)
        pred_v_sigma = self.cls_v_sigma(feats)
        pred_d_coord = self.cls_d_coord(feats)
        pred_d_sigma = self.cls_d_sigma(feats)

        pred_uvd_coord = torch.cat([pred_u_coord, pred_v_coord, pred_d_coord], dim=2)
        pred_uvd_sigma = torch.cat([pred_u_sigma, pred_v_sigma, pred_d_sigma], dim=2)

        output = edict(
            pred_uvd_coord=pred_uvd_coord,
            pred_uvd_sigma=pred_uvd_sigma,
            jts_feat=feats              # B, K, hidden
        )

        return output

    def _initialize(self):
        for m in self.cls_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

class RTMCCHeadParallel(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 input_size, 
                 in_featuremap_size, 
                 simcc_split_ratio=2.0, 
                 final_layer_kernel_size=1,
                 gau_cfg=dict(
                    hidden_dim=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    acf_fn='ReLU',
                    use_rel_bias=False,
                    pos_enc=False
                 )):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        # Difine SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2
        )
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)
        )
        self.coord_gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc']
        )
        self.sigma_gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc']            
        )

        self.cls_u_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)
        self.cls_v_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)
        self.cls_d_coord = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)
        self.cls_u_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)
        self.cls_v_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)
        self.cls_d_sigma = nn.Linear(gau_cfg['hidden_dims'], 1, bias=True)

        self.cls_head = [self.cls_u_coord, self.cls_u_sigma, self.cls_u_coord, self.cls_u_sigma, self.cls_d_coord, self.cls_d_sigma]

    def forward(self, feats):
        batch_size = feats.shape[0]

        feats = self.final_layer(feats)     # -> B, K, H, W
        
        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)             # -> B, K, hidden

        coord_feats = self.coord_gau(feats)             # -> B, K, hidden
        sigma_feats = self.sigma_gau(feats)             # -> B, K, hidden

        pred_u_coord = self.cls_u_coord(coord_feats)
        # .reshape(batch_size, 29, 1)
        pred_u_sigma = self.cls_u_sigma(sigma_feats)
        pred_v_coord = self.cls_v_coord(coord_feats)
        pred_v_sigma = self.cls_v_sigma(sigma_feats)
        pred_d_coord = self.cls_d_coord(coord_feats)
        pred_d_sigma = self.cls_d_sigma(sigma_feats)

        pred_uvd_coord = torch.cat([pred_u_coord, pred_v_coord, pred_d_coord], dim=2)
        pred_uvd_sigma = torch.cat([pred_u_sigma, pred_v_sigma, pred_d_sigma], dim=2)

        output = edict(
            pred_uvd_coord=pred_uvd_coord,
            pred_uvd_sigma=pred_uvd_sigma
        )

        return output

    def _initialize(self):
        for m in self.cls_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)    

@SPPE.register_module
class Simple3DPoseBaseSMPLCam(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLCam, self).__init__()
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

        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

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

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x, flip_test=False, **kwargs):
        ##
        # flip_test = False

        batch_size = x.shape[0]

        # start time
        # start_time = time.time()

        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)

        # posenet end time
        # posenet_end_time = time.time()
        # print('posenet inference time: {}'.format(posenet_end_time - start_time))

        # between posenet and smpl

        between_time_start = time.time()

        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_out = self.deconv_layers(flip_x0)
            flip_out = self.final_layer(flip_out)

            # flip heatmap
            flip_out = flip_out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            flip_out = self.flip_heatmap(flip_out)

            out = out.reshape((out.shape[0], self.num_joints, -1))
            flip_out = flip_out.reshape((flip_out.shape[0], self.num_joints, -1))

            heatmaps = norm_heatmap(self.norm_type, out)
            flip_heatmaps = norm_heatmap(self.norm_type, flip_out)
            heatmaps = (heatmaps + flip_heatmaps) / 2
        else:
            out = out.reshape((out.shape[0], self.num_joints, -1))

            out = norm_heatmap(self.norm_type, out)
            assert out.dim() == 3, out.shape

            heatmaps = out / out.sum(dim=2, keepdim=True)

        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

        hm_x0 = heatmaps.sum((2, 3))
        hm_y0 = heatmaps.sum((2, 4))
        hm_z0 = heatmaps.sum((3, 4))

        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
        hm_x = hm_x0 * range_tensor
        hm_y = hm_y0 * range_tensor
        hm_z = hm_z0 * range_tensor

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # shape time
        shape_time_start = time.time()

        xc = x0

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        
        # shape_time_end = time.time()
        # print('shape time: {}'.format(shape_time_end - shape_time_start))
        
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
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1),
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )


        # end time
        # end_time = time.time()
        # print('total inference time: {}'.format(end_time - start_time))

        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output

@SPPE.register_module
class Simple3DPoseBaseSMPLCamRLE(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLCamRLE, self).__init__()
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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

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

@SPPE.register_module
class SMPLCamRLESimcc(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimcc, self).__init__()
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
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            if kwargs['BACKBONE_PRETRAINED'] == 'imagenet':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'cocopose':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {}
        if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':        
            state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
            for k, v in model_state.items():
                prefix = 'backbone.'
                pre_k = prefix + k
                if pre_k in x:
                    state[k] = x[pre_k]
        else:
            raise NotImplementedError            

        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=512,
            out_channels=29,
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(batch_size, self.num_joints, 3)
        sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

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
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )
        return output

@SPPE.register_module
class SMPLCamRLESimccAddFeat(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccAddFeat, self).__init__()
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
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            if kwargs['BACKBONE_PRETRAINED'] == 'imagenet':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'cocopose':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {}
        if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':        
            state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
            for k, v in model_state.items():
                prefix = 'backbone.'
                pre_k = prefix + k
                if pre_k in x:
                    state[k] = x[pre_k]
        else:
            raise NotImplementedError            

        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=512,
            out_channels=29,
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        # add joints feat
        self.fc1 = nn.Linear(self.feature_channel + 256, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(batch_size, self.num_joints, 3)
        sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

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

        # add jts_feat
        x1 = posenet_output.jts_feat        # (B, K, hidden size)
        x1 = torch.transpose(x1, 1, 2)      # B, C, K
        x1 = self.avg_pool_1d(x1)           # B, C, 1
        x1 = x1.view(x1.size(0), -1)        # B, 256

        xc = torch.cat((x0, x1), dim=1)     # B, 512 + 256

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

            # add jts_feat
            flip_x1 = flip_posenet_output.jts_feat
            flip_x1 = torch.transpose(flip_x1, 1, 2)
            flip_x1 = self.avg_pool_1d(flip_x1)
            flip_x1 = flip_x1.view(flip_x1.size(0), -1)

            flip_xc = torch.cat((flip_x0, flip_x1), dim=1)

            flip_xc = self.fc1(flip_xc)
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
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )
        return output    

@SPPE.register_module
class SMPLCamRLESimccParallel(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccParallel, self).__init__()
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
        # RTM-Simcc Head
        self.pose_head = RTMCCHeadParallel(
            in_channels=512,
            out_channels=29,
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(batch_size, self.num_joints, 3)
        sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

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
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )
        return output    

@SPPE.register_module
class SMPLCamRLESimccShape(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccShape, self).__init__()
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
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            if kwargs['BACKBONE_PRETRAINED'] == 'imagenet':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'cocopose':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {}
        if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':        
            state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
            for k, v in model_state.items():
                prefix = 'backbone.'
                pre_k = prefix + k
                if pre_k in x:
                    state[k] = x[pre_k]
        else:
            raise NotImplementedError            

        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=512,
            out_channels=29,
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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
        self.fc1 = nn.Linear(self.feature_channel+10+3, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(batch_size, self.num_joints, 3)
        sigma = out_sigma.reshape(batch_size, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, self.num_joints, 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, self.num_joints, -1)

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

        # xc = x0
        pred_shape = init_shape
        pred_camera = init_cam
        for i in range(3):
            xc = torch.cat([x0, pred_shape, pred_camera], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_shape = self.decshape(xc).reshape(batch_size, -1) + pred_shape
            pred_camera = self.deccam(xc).reshape(batch_size, -1) + pred_camera

        # delta_shape = self.decshape(xc)
        # pred_shape = delta_shape + init_shape
        
        pred_phi = self.decphi(xc)
        # pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)
            flip_pred_shape = init_shape
            flip_pred_camera = init_cam
            for i in range(3):
                flip_xc = torch.cat([flip_x0, flip_pred_shape, flip_pred_camera], 1)
                flip_xc = self.fc1(flip_xc)
                flip_xc = self.drop1(flip_xc)
                flip_xc = self.fc2(flip_xc)
                flip_xc = self.drop2(flip_xc)
                flip_pred_shape = self.decshape(flip_xc).reshape(batch_size, -1) + flip_pred_shape
                flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + flip_pred_camera
            flip_pred_phi = self.decphi(flip_xc)
            # flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

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
            # pred_delta_shape=delta_shape,
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
            cam_scale=camScale[:, 0],
            cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            # img_feat=x0
        )
        return output    

@SPPE.register_module
class SMPLCamRLESimccVMFTLoop(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccVMFTLoop, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.num_markers = kwargs['NUM_MARKERS']
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
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {}
        if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':        
            state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
            for k, v in model_state.items():
                prefix = 'backbone.'
                pre_k = prefix + k
                if pre_k in x:
                    state[k] = x[pre_k]
        else:
            raise NotImplementedError
        
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=512,
            out_channels=(29+67),
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.marker_pairs_67 = ((0, 64), (1, 2), (3, 6), (4, 5), (7, 8),
                                (9, 10), (11,12),(13,14),(16,17),(18,19),
                                (20,21), (22,24),(25,26),(28,29),(30,31),
                                (32,33), (34,35),(36,37),(38,39),(40,41),
                                (42,43), (44,45),(46,47),(48,49),(50,51),
                                (52,53), (54,55),(56,57),(58,59),(62,63),
                                (65,66))

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
        self.avg_pool_1d_mks = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_1d_jts = nn.AdaptiveAvgPool1d(1)

        # add joints feat
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        # Finetuning Loop
        self.fc3 = nn.Linear(self.feature_channel + 256*2 + 10 + 24*4, 1024)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(1024, 1024)
        self.drop4 = nn.Dropout(p=0.5)
        self.decshape2 = nn.Linear(1024, 10)
        self.dectheta = nn.Linear(1024, 24*4)

        self.fc_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)
        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)
        # out_feats = posenet_output.jts_feat     # -> B, ( K + J ), hidden

        pred_jts = out_coord[:, :self.num_joints, :]
        pred_mks = out_coord[:, self.num_joints:, :]
        # pred_mks_feat = out_feats[:, self.num_joints:, :]

        # (B, N, 3)
        sigma = out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
        sigma_29 = sigma[:, :self.num_joints, :]
        sigma_67 = sigma[:, self.num_joints:, :]
        scores_29 = 1 - sigma_29
        scores_67 = 1 - sigma_67
        scores_29 = torch.mean(scores_29, dim=2, keepdim=True)
        scores_67 = torch.mean(scores_67, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test and not self.training:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)

            flip_pred_jts = flip_out_coord[:, :self.num_joints, :]
            flip_pred_mks = flip_out_coord[:, self.num_joints:, :]

            flip_sigma = flip_out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
            
            flip_sigma_29 = flip_sigma[:, :self.num_joints, :]
            flip_sigma_67 = flip_sigma[:, self.num_joints:, :]

            flip_scores_29 = 1 - flip_sigma_29
            flip_scores_29 = torch.mean(flip_scores_29, dim=2, keepdim=True)

            flip_scores_67 = 1 - flip_sigma_67
            flip_scores_67 = torch.mean(flip_scores_67, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_scores_29, flip_sigma_29]
            flip_pred_jts, flip_scores_29, flip_sigma_29 = flip_coord( flip_preds, 
                                                                 joint_pairs=self.joint_pairs_29,
                                                                 width_dim=self.width_dim,
                                                                 shift=True,
                                                                 flatten=False)

            flip_preds_mks = [flip_pred_mks, flip_scores_67, flip_sigma_67]
            flip_pred_mks, flip_scores_67, flip_sigma_67 = flip_coord(
                flip_preds_mks,
                joint_pairs=self.marker_pairs_67,
                width_dim=self.width_dim,
                shift=True,
                flatten=False
            )

            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            scores_29 = (scores_29 + flip_scores_29) / 2
            sigma_29 = (sigma_29 + flip_sigma_29) / 2

            pred_mks = (pred_mks + flip_pred_mks) / 2
            scores_67 = (scores_67 + flip_scores_67) / 2
            sigma_67 = (sigma_67 + flip_sigma_67) / 2

        if labels is not None and self.training:
            gt_uvd_29 = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_67 = labels['target_uvd_67'].reshape(pred_mks.shape)
            gt_uvd_weight_29 = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_uvd_weight_67 = labels['target_weight_67'].reshape(pred_mks.shape)
            # gt_3d_29_mask = gt_uvd_weight_29[:, :, 2].reshape(-1)
            # gt_3d_67_mask = gt_uvd_weight_67[:, :, 2].reshape(-1)
            gt_3d_mask = torch.cat([gt_uvd_weight_29[:, :, 2], gt_uvd_weight_67[:, :, 2]], dim=1).reshape(-1)

            assert pred_jts.shape == sigma_29.shape, (pred_jts.shape, sigma_29.shape)
            # * gt_uvd_weight
            bar_mu_29 = (pred_jts - gt_uvd_29) * gt_uvd_weight_29 / sigma_29
            bar_mu_67 = (pred_mks - gt_uvd_67) * gt_uvd_weight_67 / sigma_67
            bar_mu = torch.cat([bar_mu_29, bar_mu_67], dim=1).reshape(-1, 3)

            # bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, (self.num_joints + self.num_markers), 1)
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
        pred_uvd_mks_67 = pred_mks

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # add mks_feat
        jts_feat = posenet_output.jts_feat                    # (B, (J+K), hidden size)
        x1 = jts_feat[:, self.num_joints:, :]                 # (B, K, hidden size)
        x1 = torch.transpose(x1, 1, 2)                        # (B, C, K)
        x1 = self.avg_pool_1d_mks(x1)                             # (B, C, 1)
        x1 = x1.view(x1.size(0), -1)                          # (B, 256)

        x2 = jts_feat[:, :self.num_joints, :]                 # (B, J, hidden size)
        x2 = torch.transpose(x2, 1, 2)                        # (B, C, J)
        x2 = self.avg_pool_1d_jts(x2)                         # (B, C, 1)
        x2 = x2.view(x2.size(0), -1)                          # (B, 256)

        # xc = torch.cat((x0, x1), dim=1)                       # (B, 512 + 256)  (B, 512 + self.num_mks * 3)
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

            # add jts_feat
            # flip_jts_feat = flip_posenet_output.jts_feat
            # flip_x1 = flip_pred_mks.reshape(batch_size, -1)
            # flip_x1 = flip_jts_feat[:, self.num_joints:, :]
            # flip_x1 = torch.transpose(flip_x1, 1, 2)
            # flip_x1 = self.avg_pool_1d(flip_x1)
            # flip_x1 = flip_x1.view(flip_x1.size(0), -1)

            # flip_xc = torch.cat((flip_x0, flip_x1), dim=1)
            flip_xc = flip_x0

            flip_xc = self.fc1(flip_xc)
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
        pred_xyz_mks_67 = torch.zeros_like(pred_uvd_mks_67)
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
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m
            pred_xy_mks_67_meter = ((pred_uvd_mks_67[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_mks_67[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xy_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()            
            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m
            pred_xyz_mks_67_meter = (pred_xyz_mks_67[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_mks_67[:, :, 2:]*self.depth_factor + camDepth) - camTrans

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xyz_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_mks_67 = pred_xyz_mks_67 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_xyz_mks_67_flat = pred_xyz_mks_67.reshape(batch_size, -1)
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
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        # add fine-tuning loop
        pred_shape_ft = pred_shape
        pred_theta_mats_ft = pred_theta_mats
        for i in range(3):
            xf = torch.cat([x0, x1, x2, pred_shape_ft, pred_theta_mats_ft], dim=1)                # (B, 1024+96+10)
            xf = self.fc3(xf)
            xf = self.drop3(xf)
            xf = self.fc4(xf)
            xf = self.drop4(xf)
            pred_shape_ft = self.decshape2(xf).reshape(batch_size, -1) + pred_shape_ft
            pred_theta_mats_ft = self.dectheta(xf).reshape(batch_size, -1) + pred_theta_mats_ft
        output_ft = self.smpl(
            pose_axis_angle=pred_theta_mats_ft,
            betas = pred_shape_ft,
            global_orient=None,
            transl=None,
            pose2rot=True
        )
        pred_vertices = output_ft.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output_ft.joints.float() / self.depth_factor
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output_ft.joints_from_verts.float() / self.depth_factor
        # pred_theta_mats_ft = output_ft.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = camera_root - output_ft.joints.float().reshape(-1, 24, 3)[:, 0, :]
        
        output = edict(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_shape_ft=pred_shape_ft,
            pred_theta_mats=pred_theta_mats,
            pred_theta_mats_ft=pred_theta_mats_ft,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_uvd_mks=pred_uvd_mks_67,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_mks_67=pred_xyz_mks_67_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=scores_29.float(),
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

@SPPE.register_module
class SMPLCamRLESimccVM(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccVM, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.num_markers = kwargs['NUM_MARKERS']
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
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':
                x = tm.resnet18(pretrained=True)
                self.feature_channel = 512
            elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
                file_path = 'model_files/best_AP_epoch_210.pth'
                pretrain_dict = torch.load(file_path)
                x = pretrain_dict['state_dict']
                self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {}
        if kwargs['BACKBONE_PRETRAINED'] == 'IMAGENET':        
            state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        elif kwargs['BACKBONE_PRETRAINED'] == 'COCOPOSE':
            for k, v in model_state.items():
                prefix = 'backbone.'
                pre_k = prefix + k
                if pre_k in x:
                    state[k] = x[pre_k]
        else:
            raise NotImplementedError
        
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=512,
            out_channels=(29+67),
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.marker_pairs_67 = ((0, 64), (1, 2), (3, 6), (4, 5), (7, 8),
                                (9, 10), (11,12),(13,14),(16,17),(18,19),
                                (20,21), (22,24),(25,26),(28,29),(30,31),
                                (32,33), (34,35),(36,37),(38,39),(40,41),
                                (42,43), (44,45),(46,47),(48,49),(50,51),
                                (52,53), (54,55),(56,57),(58,59),(62,63),
                                (65,66))

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
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)

        # add joints feat
        self.fc1 = nn.Linear(self.feature_channel + 256, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)
        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)
        # out_feats = posenet_output.jts_feat     # -> B, ( K + J ), hidden

        pred_jts = out_coord[:, :self.num_joints, :]
        pred_mks = out_coord[:, self.num_joints:, :]
        # pred_mks_feat = out_feats[:, self.num_joints:, :]

        # (B, N, 3)
        sigma = out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
        sigma_29 = sigma[:, :self.num_joints, :]
        sigma_67 = sigma[:, self.num_joints:, :]
        scores_29 = 1 - sigma_29
        scores_67 = 1 - sigma_67
        scores_29 = torch.mean(scores_29, dim=2, keepdim=True)
        scores_67 = torch.mean(scores_67, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test and not self.training:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)

            flip_pred_jts = flip_out_coord[:, :self.num_joints, :]
            flip_pred_mks = flip_out_coord[:, self.num_joints:, :]

            flip_sigma = flip_out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
            
            flip_sigma_29 = flip_sigma[:, :self.num_joints, :]
            flip_sigma_67 = flip_sigma[:, self.num_joints:, :]

            flip_scores_29 = 1 - flip_sigma_29
            flip_scores_29 = torch.mean(flip_scores_29, dim=2, keepdim=True)

            flip_scores_67 = 1 - flip_sigma_67
            flip_scores_67 = torch.mean(flip_scores_67, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_scores_29, flip_sigma_29]
            flip_pred_jts, flip_scores_29, flip_sigma_29 = flip_coord( flip_preds, 
                                                                 joint_pairs=self.joint_pairs_29,
                                                                 width_dim=self.width_dim,
                                                                 shift=True,
                                                                 flatten=False)

            flip_preds_mks = [flip_pred_mks, flip_scores_67, flip_sigma_67]
            flip_pred_mks, flip_scores_67, flip_sigma_67 = flip_coord(
                flip_preds_mks,
                joint_pairs=self.marker_pairs_67,
                width_dim=self.width_dim,
                shift=True,
                flatten=False
            )

            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            scores_29 = (scores_29 + flip_scores_29) / 2
            sigma_29 = (sigma_29 + flip_sigma_29) / 2

            pred_mks = (pred_mks + flip_pred_mks) / 2
            scores_67 = (scores_67 + flip_scores_67) / 2
            sigma_67 = (sigma_67 + flip_sigma_67) / 2

        if labels is not None and self.training:
            gt_uvd_29 = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_67 = labels['target_uvd_67'].reshape(pred_mks.shape)
            gt_uvd_weight_29 = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_uvd_weight_67 = labels['target_weight_67'].reshape(pred_mks.shape)
            # gt_3d_29_mask = gt_uvd_weight_29[:, :, 2].reshape(-1)
            # gt_3d_67_mask = gt_uvd_weight_67[:, :, 2].reshape(-1)
            gt_3d_mask = torch.cat([gt_uvd_weight_29[:, :, 2], gt_uvd_weight_67[:, :, 2]], dim=1).reshape(-1)

            assert pred_jts.shape == sigma_29.shape, (pred_jts.shape, sigma_29.shape)
            # * gt_uvd_weight
            bar_mu_29 = (pred_jts - gt_uvd_29) * gt_uvd_weight_29 / sigma_29
            bar_mu_67 = (pred_mks - gt_uvd_67) * gt_uvd_weight_67 / sigma_67
            bar_mu = torch.cat([bar_mu_29, bar_mu_67], dim=1).reshape(-1, 3)

            # bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, (self.num_joints + self.num_markers), 1)
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
        pred_uvd_mks_67 = pred_mks

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # add mks_feat
        jts_feat = posenet_output.jts_feat                    # (B, (J+K), hidden size)
        # x1 = pred_mks.reshape(batch_size, -1)                   # (B, self.num_mks * 3)
        x1 = jts_feat[:, self.num_joints:, :]                 # (B, K, hidden size)
        x1 = torch.transpose(x1, 1, 2)                        # (B, C, K)
        x1 = self.avg_pool_1d(x1)                             # (B, C, 1)
        x1 = x1.view(x1.size(0), -1)                          # (B, 256)

        xc = torch.cat([x0, x1], dim=1)                       # (B, 512 + 256)  (B, 512 + self.num_mks * 3)
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        pred_shape = self.decshape(xc).reshape(batch_size, -1) + init_shape
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        pred_phi = self.decphi(xc)
        # pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)

            # add jts_feat
            flip_jts_feat = flip_posenet_output.jts_feat
            # flip_x1 = flip_pred_mks.reshape(batch_size, -1)
            flip_x1 = flip_jts_feat[:, self.num_joints:, :]
            flip_x1 = torch.transpose(flip_x1, 1, 2)
            flip_x1 = self.avg_pool_1d(flip_x1)
            flip_x1 = flip_x1.view(flip_x1.size(0), -1)

            flip_xc = torch.cat([flip_x0, flip_x1], 1)
            flip_xc = self.fc1(flip_xc)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)
            flip_pred_shape = self.decshape(flip_xc).reshape(batch_size, -1) + init_shape
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam
            flip_pred_phi = self.decphi(flip_xc)
            # flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

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
        pred_xyz_mks_67 = torch.zeros_like(pred_uvd_mks_67)
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
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m
            pred_xy_mks_67_meter = ((pred_uvd_mks_67[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_mks_67[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xy_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()            
            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m
            pred_xyz_mks_67_meter = (pred_xyz_mks_67[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_mks_67[:, :, 2:]*self.depth_factor + camDepth) - camTrans

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xyz_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_mks_67 = pred_xyz_mks_67 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_xyz_mks_67_flat = pred_xyz_mks_67.reshape(batch_size, -1)
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
            # pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_uvd_mks=pred_uvd_mks_67,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_mks_67=pred_xyz_mks_67_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=scores_29.float(),
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

@SPPE.register_module
class SMPLCamRLESimccVMHrnet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccVMHrnet, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.num_markers = kwargs['NUM_MARKERS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        # HRNet
        hrnet_cfg_file = 'configs/hrnetw64_cfg.yaml'
        with open(hrnet_cfg_file) as f:
            hrnet_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

        backbone = HRNet(cfg=hrnet_cfg)

        self.preact = backbone
        self.feature_channel = 2048

        # COCO pretrain
        # file_path = 'model_files/best_coco_AP_epoch_160.pth'
        file_path = 'model_files/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'
        pretrain_dict = torch.load(file_path)
        pretrain_dict = pretrain_dict['state_dict']

        model_state = self.preact.state_dict()
        state = {}
        for k, v in model_state.items():
            prefix = 'backbone.'
            pre_k = prefix + k
            if pre_k in pretrain_dict:
                state[k] = pretrain_dict[pre_k]
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=2048,
            out_channels=(29+67),
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.marker_pairs_67 = ((0, 64), (1, 2), (3, 6), (4, 5), (7, 8),
                                (9, 10), (11,12),(13,14),(16,17),(18,19),
                                (20,21), (22,24),(25,26),(28,29),(30,31),
                                (32,33), (34,35),(36,37),(38,39),(40,41),
                                (42,43), (44,45),(46,47),(48,49),(50,51),
                                (52,53), (54,55),(56,57),(58,59),(62,63),
                                (65,66))

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
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)

        # add joints feat
        self.fc1 = nn.Linear(self.feature_channel + 256, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.fc_layers = [self.fc1, self.fc2]

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)
        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)
        # out_feats = posenet_output.jts_feat     # -> B, ( K + J ), hidden

        pred_jts = out_coord[:, :self.num_joints, :]
        pred_mks = out_coord[:, self.num_joints:, :]
        # pred_mks_feat = out_feats[:, self.num_joints:, :]

        # (B, N, 3)
        sigma = out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
        sigma_29 = sigma[:, :self.num_joints, :]
        sigma_67 = sigma[:, self.num_joints:, :]
        scores_29 = 1 - sigma_29
        scores_67 = 1 - sigma_67
        scores_29 = torch.mean(scores_29, dim=2, keepdim=True)
        scores_67 = torch.mean(scores_67, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test and not self.training:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)

            flip_pred_jts = flip_out_coord[:, :self.num_joints, :]
            flip_pred_mks = flip_out_coord[:, self.num_joints:, :]

            flip_sigma = flip_out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
            
            flip_sigma_29 = flip_sigma[:, :self.num_joints, :]
            flip_sigma_67 = flip_sigma[:, self.num_joints:, :]

            flip_scores_29 = 1 - flip_sigma_29
            flip_scores_29 = torch.mean(flip_scores_29, dim=2, keepdim=True)

            flip_scores_67 = 1 - flip_sigma_67
            flip_scores_67 = torch.mean(flip_scores_67, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_scores_29, flip_sigma_29]
            flip_pred_jts, flip_scores_29, flip_sigma_29 = flip_coord( flip_preds, 
                                                                 joint_pairs=self.joint_pairs_29,
                                                                 width_dim=self.width_dim,
                                                                 shift=True,
                                                                 flatten=False)

            flip_preds_mks = [flip_pred_mks, flip_scores_67, flip_sigma_67]
            flip_pred_mks, flip_scores_67, flip_sigma_67 = flip_coord(
                flip_preds_mks,
                joint_pairs=self.marker_pairs_67,
                width_dim=self.width_dim,
                shift=True,
                flatten=False
            )

            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            scores_29 = (scores_29 + flip_scores_29) / 2
            sigma_29 = (sigma_29 + flip_sigma_29) / 2

            pred_mks = (pred_mks + flip_pred_mks) / 2
            scores_67 = (scores_67 + flip_scores_67) / 2
            sigma_67 = (sigma_67 + flip_sigma_67) / 2

        if labels is not None and self.training:
            gt_uvd_29 = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_67 = labels['target_uvd_67'].reshape(pred_mks.shape)
            gt_uvd_weight_29 = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_uvd_weight_67 = labels['target_weight_67'].reshape(pred_mks.shape)
            # gt_3d_29_mask = gt_uvd_weight_29[:, :, 2].reshape(-1)
            # gt_3d_67_mask = gt_uvd_weight_67[:, :, 2].reshape(-1)
            gt_3d_mask = torch.cat([gt_uvd_weight_29[:, :, 2], gt_uvd_weight_67[:, :, 2]], dim=1).reshape(-1)

            assert pred_jts.shape == sigma_29.shape, (pred_jts.shape, sigma_29.shape)
            # * gt_uvd_weight
            bar_mu_29 = (pred_jts - gt_uvd_29) * gt_uvd_weight_29 / sigma_29
            bar_mu_67 = (pred_mks - gt_uvd_67) * gt_uvd_weight_67 / sigma_67
            bar_mu = torch.cat([bar_mu_29, bar_mu_67], dim=1).reshape(-1, 3)

            # bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, (self.num_joints + self.num_markers), 1)
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
        pred_uvd_mks_67 = pred_mks

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # add mks_feat
        jts_feat = posenet_output.jts_feat                    # (B, (J+K), hidden size)
        # x1 = pred_mks.reshape(batch_size, -1)               # (B, self.num_mks * 3)
        x1 = jts_feat[:, self.num_joints:, :]                 # (B, K, hidden size)
        x1 = torch.transpose(x1, 1, 2)                        # (B, C, K)
        x1 = self.avg_pool_1d(x1)                             # (B, C, 1)
        x1 = x1.view(x1.size(0), -1)                          # (B, 256)

        xc = torch.cat([x0, x1], dim=1)                       # (B, 512 + 256)  (B, 512 + self.num_mks * 3)
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        pred_shape = self.decshape(xc).reshape(batch_size, -1) + init_shape
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        pred_phi = self.decphi(xc)
        # pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)

            # add jts_feat
            flip_jts_feat = flip_posenet_output.jts_feat
            # flip_x1 = flip_pred_mks.reshape(batch_size, -1)
            flip_x1 = flip_jts_feat[:, self.num_joints:, :]
            flip_x1 = torch.transpose(flip_x1, 1, 2)
            flip_x1 = self.avg_pool_1d(flip_x1)
            flip_x1 = flip_x1.view(flip_x1.size(0), -1)

            flip_xc = torch.cat([flip_x0, flip_x1], 1)
            flip_xc = self.fc1(flip_xc)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)
            flip_pred_shape = self.decshape(flip_xc).reshape(batch_size, -1) + init_shape
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam
            flip_pred_phi = self.decphi(flip_xc)
            # flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

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
        pred_xyz_mks_67 = torch.zeros_like(pred_uvd_mks_67)
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
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m
            pred_xy_mks_67_meter = ((pred_uvd_mks_67[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_mks_67[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xy_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()            
            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m
            pred_xyz_mks_67_meter = (pred_xyz_mks_67[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_mks_67[:, :, 2:]*self.depth_factor + camDepth) - camTrans

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xyz_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_mks_67 = pred_xyz_mks_67 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_xyz_mks_67_flat = pred_xyz_mks_67.reshape(batch_size, -1)
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
            # pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_uvd_mks=pred_uvd_mks_67,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_mks_67=pred_xyz_mks_67_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=scores_29.float(),
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

@SPPE.register_module
class SMPLCamRLESimccVMMGFENet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SMPLCamRLESimccVMMGFENet, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.num_markers = kwargs['NUM_MARKERS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        # # MGFENet
        # mgfenet_cfg_file = 'configs/hrnetw64_cfg.yaml'
        # with open(hrnet_cfg_file) as f:
        #     hrnet_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

        # MGFENet
        backbone = MGFENet(
            arch='big_7x7',
            out_indices=(-1,),
        )

        self.preact = backbone
        self.feature_channel = 960

        # # COCO pretrain
        file_path = 'model_files/MGFENet_Pretrain.pth'
        pretrain_dict = torch.load(file_path)
        pretrain_dict = pretrain_dict['state_dict']

        model_state = self.preact.state_dict()
        state = {}
        for k, v in model_state.items():
            prefix = 'backbone.'
            pre_k = prefix + k
            if pre_k in pretrain_dict:
                state[k] = pretrain_dict[pre_k]
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # Posenet Head
        # RTM-Simcc Head
        self.pose_head = RTMCCHead(
            in_channels=960,
            out_channels=(29+67),
            input_size=(256, 256),
            in_featuremap_size=(8, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False
            )
        )
        self.pose_head._initialize()

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

        self.marker_pairs_67 = ((0, 64), (1, 2), (3, 6), (4, 5), (7, 8),
                                (9, 10), (11,12),(13,14),(16,17),(18,19),
                                (20,21), (22,24),(25,26),(28,29),(30,31),
                                (32,33), (34,35),(36,37),(38,39),(40,41),
                                (42,43), (44,45),(46,47),(48,49),(50,51),
                                (52,53), (54,55),(56,57),(58,59),(62,63),
                                (65,66))

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
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)

        # add joints feat
        self.fc1 = nn.Linear(self.feature_channel + 256, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 3)

        self.fc_layers = [self.fc1, self.fc2]

        self.focal_length = kwargs['FOCAL_LENGTH']
        self.bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.depth_factor = float(self.bbox_3d_shape[2]) * 1e-3
        self.input_size = 256.0

        # self._load_flow_dict()
        

    def _load_flow_dict(self):
        file_path = 'model_files/litehuman_ckpt/vm_hrnetw48.pth'
        pretrain_dict = torch.load(file_path)
        state = {}
        for k, v in pretrain_dict.items():
            t = k.split('.')
            if t[0] == 'flow2d' or t[0] == 'flow3d':
                state[k] = v
        model_state = self.state_dict()
        model_state.update(state)
        self.load_state_dict(model_state)
        self.flow2d.requires_grad_(False)
        self.flow3d.requires_grad_(False)
        

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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)
        x0 = x0[-1]

        # pose net
        posenet_output = self.pose_head(x0)

        out_coord = posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)
        out_sigma = posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)
        # out_feats = posenet_output.jts_feat     # -> B, ( K + J ), hidden

        pred_jts = out_coord[:, :self.num_joints, :]
        pred_mks = out_coord[:, self.num_joints:, :]
        # pred_mks_feat = out_feats[:, self.num_joints:, :]

        # (B, N, 3)
        sigma = out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
        sigma_29 = sigma[:, :self.num_joints, :]
        sigma_67 = sigma[:, self.num_joints:, :]
        scores_29 = 1 - sigma_29
        scores_67 = 1 - sigma_67
        scores_29 = torch.mean(scores_29, dim=2, keepdim=True)
        scores_67 = torch.mean(scores_67, dim=2, keepdim=True)

        # flip test, compute flipped pred_jts, scores, sigma
        if flip_test and not self.training:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_x0 = flip_x0[-1]
            # pose net
            flip_posenet_output = self.pose_head(flip_x0)

            flip_out_coord = flip_posenet_output.pred_uvd_coord.reshape(batch_size, (self.num_joints + self.num_markers), 3)

            flip_out_sigma = flip_posenet_output.pred_uvd_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1)

            flip_pred_jts = flip_out_coord[:, :self.num_joints, :]
            flip_pred_mks = flip_out_coord[:, self.num_joints:, :]

            flip_sigma = flip_out_sigma.reshape(batch_size, (self.num_joints + self.num_markers), -1).sigmoid() + 1e-9
            
            flip_sigma_29 = flip_sigma[:, :self.num_joints, :]
            flip_sigma_67 = flip_sigma[:, self.num_joints:, :]

            flip_scores_29 = 1 - flip_sigma_29
            flip_scores_29 = torch.mean(flip_scores_29, dim=2, keepdim=True)

            flip_scores_67 = 1 - flip_sigma_67
            flip_scores_67 = torch.mean(flip_scores_67, dim=2, keepdim=True)

            # flip pred back
            flip_preds = [flip_pred_jts, flip_scores_29, flip_sigma_29]
            flip_pred_jts, flip_scores_29, flip_sigma_29 = flip_coord( flip_preds, 
                                                                 joint_pairs=self.joint_pairs_29,
                                                                 width_dim=self.width_dim,
                                                                 shift=True,
                                                                 flatten=False)

            flip_preds_mks = [flip_pred_mks, flip_scores_67, flip_sigma_67]
            flip_pred_mks, flip_scores_67, flip_sigma_67 = flip_coord(
                flip_preds_mks,
                joint_pairs=self.marker_pairs_67,
                width_dim=self.width_dim,
                shift=True,
                flatten=False
            )

            # average
            pred_jts = (pred_jts + flip_pred_jts) / 2
            scores_29 = (scores_29 + flip_scores_29) / 2
            sigma_29 = (sigma_29 + flip_sigma_29) / 2

            pred_mks = (pred_mks + flip_pred_mks) / 2
            scores_67 = (scores_67 + flip_scores_67) / 2
            sigma_67 = (sigma_67 + flip_sigma_67) / 2

        if labels is not None and self.training:
            gt_uvd_29 = labels['target_uvd_29'].reshape(pred_jts.shape)
            gt_uvd_67 = labels['target_uvd_67'].reshape(pred_mks.shape)
            gt_uvd_weight_29 = labels['target_weight_29'].reshape(pred_jts.shape)
            gt_uvd_weight_67 = labels['target_weight_67'].reshape(pred_mks.shape)
            # gt_3d_29_mask = gt_uvd_weight_29[:, :, 2].reshape(-1)
            # gt_3d_67_mask = gt_uvd_weight_67[:, :, 2].reshape(-1)
            gt_3d_mask = torch.cat([gt_uvd_weight_29[:, :, 2], gt_uvd_weight_67[:, :, 2]], dim=1).reshape(-1)

            assert pred_jts.shape == sigma_29.shape, (pred_jts.shape, sigma_29.shape)
            # * gt_uvd_weight
            bar_mu_29 = (pred_jts - gt_uvd_29) * gt_uvd_weight_29 / sigma_29
            bar_mu_67 = (pred_mks - gt_uvd_67) * gt_uvd_weight_67 / sigma_67
            bar_mu = torch.cat([bar_mu_29, bar_mu_67], dim=1).reshape(-1, 3)

            # bar_mu = (pred_jts - gt_uvd) * gt_uvd_weight / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(batch_size, (self.num_joints + self.num_markers), 1)
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
        pred_uvd_mks_67 = pred_mks

        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1) # (B, 3,)

        # add mks_feat
        jts_feat = posenet_output.jts_feat                    # (B, (J+K), hidden size)
        # x1 = pred_mks.reshape(batch_size, -1)               # (B, self.num_mks * 3)
        x1 = jts_feat[:, self.num_joints:, :]                 # (B, K, hidden size)
        x1 = torch.transpose(x1, 1, 2)                        # (B, C, K)
        x1 = self.avg_pool_1d(x1)                             # (B, C, 1)
        x1 = x1.view(x1.size(0), -1)                          # (B, 256)

        xc = torch.cat([x0, x1], dim=1)                       # (B, 512 + 256)  (B, 512 + self.num_mks * 3)
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        pred_shape = self.decshape(xc).reshape(batch_size, -1) + init_shape
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        pred_phi = self.decphi(xc)
        # pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)

            # add jts_feat
            flip_jts_feat = flip_posenet_output.jts_feat
            # flip_x1 = flip_pred_mks.reshape(batch_size, -1)
            flip_x1 = flip_jts_feat[:, self.num_joints:, :]
            flip_x1 = torch.transpose(flip_x1, 1, 2)
            flip_x1 = self.avg_pool_1d(flip_x1)
            flip_x1 = flip_x1.view(flip_x1.size(0), -1)

            flip_xc = torch.cat([flip_x0, flip_x1], 1)
            flip_xc = self.fc1(flip_xc)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)
            flip_pred_shape = self.decshape(flip_xc).reshape(batch_size, -1) + init_shape
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam
            flip_pred_phi = self.decphi(flip_xc)
            # flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam

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
        pred_xyz_mks_67 = torch.zeros_like(pred_uvd_mks_67)
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
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m
            pred_xy_mks_67_meter = ((pred_uvd_mks_67[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_mks_67[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xy_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, 2:] = pred_uvd_mks_67[:, :, 2:].clone()            
            pred_xyz_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_jts_29[:, :, 2:]*self.depth_factor + camDepth) - camTrans  # unit: m
            pred_xyz_mks_67_meter = (pred_xyz_mks_67[:, :, :2] * self.input_size / self.focal_length) \
                                            * (pred_xyz_mks_67[:, :, 2:]*self.depth_factor + camDepth) - camTrans

            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)
            pred_xyz_mks_67[:, :, :2] = pred_xyz_mks_67_meter / self.depth_factor

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_mks_67 = pred_xyz_mks_67 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_xyz_mks_67_flat = pred_xyz_mks_67.reshape(batch_size, -1)
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
            # pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29,
            pred_uvd_mks=pred_uvd_mks_67,
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_mks_67=pred_xyz_mks_67_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=scores_29.float(),
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