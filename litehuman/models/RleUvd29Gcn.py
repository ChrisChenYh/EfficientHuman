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

import time

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

class Graph:
    def __init__(self, neighbor_link=None, layout='uvd29', strategy='undirected', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.neighbor_link = neighbor_link

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A
    
    def get_edge(self, layout):
        if layout == 'uvd29':
            self.num_node = 29
            self_link = [(i, i) for i in range(self.num_node)]
            # follow HybrIK
            neighbor_link = [ 
                (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), # 5
                (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), # 11
                (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), # 17
                (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24), # 23
                (22, 25), (23, 26), (10, 27), (11, 28) # 27
            ]
            self.edge = self_link + neighbor_link
            self.source_nodes = [node[0] for node in neighbor_link]
            self.target_nodes = [node[1] for node in neighbor_link]
            self.center = 0
        if layout == 'uvd54':
            self.num_node = 54
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = self.neighbor_link
            self.edge = self_link + neighbor_link
            self.source_nodes = [node[0] for node in neighbor_link]
            self.target_nodes = [node[1] for node in neighbor_link]
            self.center = 0
    
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            # print(hop)
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)

        A = np.zeros((self.num_node, self.num_node))
        A = normalize_adjacency
        self.A = A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    
    # compute hop steps:
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    D1 = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if D1[i] > 0:
            Dn[i, i] = D1[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    D1 = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if D1[i] > 0:
            Dn[i, i] = D1[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

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
class RleUvd29Gcn(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(RleUvd29Gcn, self).__init__()
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
        # self.fc_coord = Linear(self.feature_channel, self.num_joints * 3)
        # self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 3)
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

        ##### Graph CNN Layers (Posenet uvd29)
        self.gc_feature_channels = 512
        self.gcn_layer_num = kwargs['GCN_LAYER_NUM']
        # make graph adj
        graph_uvd29 = Graph(layout='uvd29')
        adj29 = graph_uvd29.A
        adj29 = torch.from_numpy(adj29).float()
        adj29 = adj29.to_sparse()
        self.adj29 = adj29.to(torch.device('cuda'))
        gcn_layers = self._make_graphcnn_layer(A=self.adj29, input_channels=self.feature_channel, num_layers=self.gcn_layer_num, num_channels=self.gc_feature_channels)
        self.gc_encoder = nn.Sequential(*gcn_layers)
        self.uvd29_coord = nn.Sequential( GraphResBlock(self.gc_feature_channels, 64, self.adj29),
                                          GraphResBlock(64, 32, self.adj29),
                                          nn.GroupNorm(32 // 8, 32),
                                          nn.ReLU(inplace=True),
                                          GraphLinear(32, 3))
        self.uvd29_sigma = nn.Sequential( GraphResBlock(self.gc_feature_channels, 64, self.adj29),
                                          GraphResBlock(64, 32, self.adj29),
                                          nn.GroupNorm(32 // 8, 32),
                                          nn.ReLU(inplace=True),
                                          GraphLinear(32, 3))
        

    def _initialize(self):
        # for m in self.fc_layers:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=0.01)
        pass

    def _make_graphcnn_layer(self, A, input_channels=512, num_layers=5, num_channels=512):
        layers = [GraphLinear(input_channels, 2 * num_channels)]
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

    def forward(self, x, labels=None, flip_test=False, **kwargs):

        batch_size = x.shape[0]

        x0 = self.preact(x)

        # pose net
        feat = self.avg_pool_0(x0).reshape(batch_size, -1)

        ## use gcn to replace the mlp
        # 1. expend the feature map [B, 512, 1] -> [B, 512, 29]
        feat_g = feat.view(batch_size, self.feature_channel, 1).expand(-1, -1, self.num_joints)
        # 2. GCN encoders (n layers ResGCN)
        feat_g = self.gc_encoder(feat_g)
        # 3. predict coord and sigma of uvd29
        out_coord = self.uvd29_coord(feat_g)
        # print(out_coord.shape)
        # assert out_coord.shape[2] == 3
        out_sigma = self.uvd29_sigma(feat_g)

        out_coord = out_coord.permute(0, 2, 1)
        out_sigma = out_sigma.permute(0, 2, 1)

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
        shape_time_start = time.time()

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
