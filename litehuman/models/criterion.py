import torch
import torch.nn as nn

from .builder import LOSS
import math
from easydict import EasyDict as edict
from ..utils.transforms import quat_to_rotmat, rotmat_to_quat_numpy

def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()

def vertices_weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()

@LOSS.register_module
class L1LossDimSMPL(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPL, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        return loss

@LOSS.register_module
class L1LossDimSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]

        loss_uvd = weighted_l1_loss(
            pred_uvd.reshape(batch_size, -1),
            target_uvd.reshape(batch_size, -1),
            target_uvd_weight.reshape(batch_size, -1), self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        # if epoch_num > self.pretrain_epoch:
        #     loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        return loss

@LOSS.register_module
class L1LossPosenet(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossPosenet, self).__init__()
        self.elements = ELEMENTS
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.size_average = size_average
        self.reduce = reduce
    def forward(self, output, labels):
        # Joints Loss
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)

        loss = loss_uvd * self.uvd24_weight
        return loss

@LOSS.register_module
class L1LossSMPLCamRLE(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLCamRLE, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = self.elements['PRETRAIN_EPOCH']
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_loss

        # pred_jts = output.pred_uvd_jts
        sigma = output.sigma
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'].reshape(pred_uvd.shape)
        target_uvd_weight = labels['target_weight_29'].reshape(pred_uvd.shape)
        nf_loss = nf_loss * target_uvd_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            loss_uvd = nf_loss + Q_logprob
        if self.size_average and target_uvd_weight.sum() > 0:
            # loss_uvd =  loss_uvd.sum() / len(loss_uvd)
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_
            loss_uvd =  loss_uvd.sum() / loss_used_lenth

        else:
            loss_uvd = loss_uvd.sum()

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_trans=trans_loss,
            loss_scale=scale_loss
        )

        return loss_output

@LOSS.register_module
class L1LossSMPLRLE(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLRLE, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        # target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        # loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        # batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_loss
        log_sigma = output.log_sigma
        log_phi = output.log_phi

        log_phi_2d = output.log_phi_2d
        log_phi_3d = output.log_phi_3d
        # pred_jts = output.pred_uvd_jts
        sigma = output.sigma
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'].reshape(pred_uvd.shape)
        target_uvd_weight = labels['target_weight_29'].reshape(pred_uvd.shape)
        nf_loss = nf_loss * target_uvd_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            loss_uvd = nf_loss + Q_logprob
        if self.size_average and target_uvd_weight.sum() > 0:
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_

            loss_phi_3d = log_phi_3d.sum() / len(log_phi_3d)
            loss_phi_2d = log_phi_2d.sum() / len(log_phi_2d)
            loss_Q = Q_logprob.sum() / len(Q_logprob)
            loss_log_sigma = log_sigma.sum() / len(log_sigma)
            loss_phi = log_phi.sum() / len(log_phi)
            loss_sigma = sigma.sum() / len(sigma)

            loss_uvd =  loss_uvd.sum() / loss_used_lenth

            # self.global_loss_count = self.global_loss_count + 1
            # self.global_avg_uvd_loss = (self.global_avg_uvd_loss + loss_uvd) / self.global_loss_count

            # loss_uvd =  loss_uvd.sum() / len(loss_uvd)
        else:
            loss_phi_3d = log_phi_3d.sum()
            loss_phi_2d = log_phi_2d.sum()

            loss_uvd = loss_uvd.sum()
            loss_Q = Q_logprob.sum()
            loss_log_sigma = log_sigma.sum()
            loss_phi = log_phi.sum()
            loss_sigma = sigma.sum()
        # loss_uvd = weighted_l1_loss(
        #     pred_uvd.reshape(batch_size, -1),
        #     target_uvd.reshape(batch_size, -1),
        #     target_uvd_weight.reshape(batch_size, -1), self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_used_lenth=loss_used_lenth
        )

        return loss_output    

@LOSS.register_module
class L1LossSMPLCamGC(nn.Module):
    """
    Graph CNN Loss
    L_vertices_sub4 (L1 Loss) + L_beta (MSE Loss)
    """
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLCamGC, self).__init__()
        self.elements = ELEMENTS
        self.cal_beta_weight = self.elements['CAL_BETA_WEIGHT']
        self.vertices_sub4_weight = self.elements['VERTICES_SUB4_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce
    def forward(self, output, target_vertices_sub4, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']
        target_vertices_sub4_weight = labels['target_weight_vertices_sub4']
        
        loss_cal_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        
        pred_vertices_sub4 = output.pred_vertices_sub4
        batch_size = pred_vertices_sub4.shape[0]
        loss_vertices_sub4 = vertices_weighted_l1_loss( input=pred_vertices_sub4.reshape(batch_size, -1),
                                                        target=target_vertices_sub4.reshape(batch_size, -1),
                                                        weights=target_vertices_sub4_weight,
                                                        size_average=self.size_average)

        loss = loss_cal_beta * self.cal_beta_weight + loss_vertices_sub4 * self.vertices_sub4_weight
        
        loss_output = edict(
            loss=loss,
            loss_cal_beta=loss_cal_beta,
            loss_vertices_sub4=loss_vertices_sub4
        )
        return loss_output

@LOSS.register_module
class L1LossSMPLCamRLEUvd54(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLCamRLEUvd54, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.uvd54_weight = self.elements['UVD54_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, target_uvd_54, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']
        target_uvd54_weight = labels['target_weight_vertices_sub4']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_loss
        nf_uvd54_loss = output.nf_uvd54_loss

        # pred_jts = output.pred_uvd_jts
        sigma = output.sigma
        uvd54_sigma = output.uvd54_sigma

        pred_uvd = output.pred_uvd_jts
        pred_uvd54 = output.pred_uvd54_verts

        target_uvd = labels['target_uvd_29'].reshape(pred_uvd.shape)
        target_uvd_weight = labels['target_weight_29'].reshape(pred_uvd.shape)
        
        target_uvd54 = target_uvd_54.reshape(pred_uvd54.shape)
        target_uvd54_weight = target_uvd54_weight.reshape(pred_uvd54.shape)
        
        nf_loss = nf_loss * target_uvd_weight
        nf_uvd54_loss = nf_uvd54_loss * target_uvd54_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            Q_logprob_uvd54 = self.logQ(target_uvd54, pred_uvd54, uvd54_sigma) * target_uvd54_weight
            loss_uvd = nf_loss + Q_logprob
            loss_uvd54 = nf_uvd54_loss + Q_logprob_uvd54
        if self.size_average and target_uvd_weight.sum() > 0:
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_
            loss_uvd =  loss_uvd.sum() / loss_used_lenth

            loss_uvd54 = loss_uvd54.sum() / target_uvd54_weight.sum()
        else:
            loss_uvd = loss_uvd.sum()
            loss_uvd54 = loss_uvd54.sum()

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight
        loss += loss_uvd54 * self.uvd54_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_uvd54=loss_uvd54,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_used_lenth=loss_used_lenth
        )

        return loss_output

@LOSS.register_module
class L1LossSMPL54(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPL54, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.uvd54_weight = self.elements['UVD54_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, target_uvd_54, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']
        target_uvd54_weight = labels['target_weight_vertices_sub4']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_jts_loss
        nf_uvd54_loss = output.nf_verts_loss

        # pred_jts = output.pred_uvd_jts
        sigma = output.jts_sigma
        uvd54_sigma = output.verts_sigma

        pred_uvd = output.pred_uvd_jts
        pred_uvd54 = output.pred_uvd54_verts

        target_uvd = labels['target_uvd_29'].reshape(pred_uvd.shape)
        target_uvd_weight = labels['target_weight_29'].reshape(pred_uvd.shape)
        
        target_uvd54 = target_uvd_54.reshape(pred_uvd54.shape)
        target_uvd54_weight = target_uvd54_weight.reshape(pred_uvd54.shape)
        
        nf_loss = nf_loss * target_uvd_weight
        nf_uvd54_loss = nf_uvd54_loss * target_uvd54_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            Q_logprob_uvd54 = self.logQ(target_uvd54, pred_uvd54, uvd54_sigma) * target_uvd54_weight
            loss_uvd = nf_loss + Q_logprob
            loss_uvd54 = nf_uvd54_loss + Q_logprob_uvd54
        if self.size_average and target_uvd_weight.sum() > 0:
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_
            loss_uvd =  loss_uvd.sum() / loss_used_lenth

            loss_uvd54 = loss_uvd54.sum() / target_uvd54_weight.sum()
        else:
            loss_uvd = loss_uvd.sum()
            loss_uvd54 = loss_uvd54.sum()

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight
        loss += loss_uvd54 * self.uvd54_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_uvd54=loss_uvd54,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_used_lenth=loss_used_lenth,
            loss_cam = trans_loss + scale_loss
        )

        return loss_output

@LOSS.register_module
class L1LossSMPLRegressor(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLRegressor, self).__init__()
        self.elements = ELEMENTS
        self.beta_weight = self.elements['BETA_WEIGHT']
        self.theta_weight = self.elements['THETA_WEIGHT']
        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce
    
    def forward(self, output, target):
        smpl_weight = target['target_smpl_weight']
        # smpl_theta_weight = target['target_theta_weight']
        # SMPL params
        # Output: pred_theta_rotmat, pred_shape
        # pred_theta_rotmat => pred_theta_mats
        pred_theta_rotmat = output.pred_theta_rotmat
        # gt_theta_mats = target['target_theta']
        # gt_theta_rotmat = quat_to_rotmat(gt_theta_mats)
        gt_theta_rotmat = target['target_theta_rotmat']

        pred_shape = output.pred_shape
        batch_size = pred_shape.shape[0]
        pred_theta_rotmat = pred_theta_rotmat.reshape(batch_size, -1)
        gt_theta_rotmat = gt_theta_rotmat.reshape(batch_size, -1)
        loss_beta = self.criterion_smpl(pred_shape * smpl_weight, target['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(pred_theta_rotmat * smpl_weight, gt_theta_rotmat * smpl_weight)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight

        loss_output = edict(
            loss = loss,
            loss_beta = loss_beta,
            loss_theta = loss_theta
        )

        return loss_output


@LOSS.register_module
class L1LossSMPLCamVMRLE(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLCamVMRLE, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz67_weight = self.elements['XYZ67_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = self.elements['PRETRAIN_EPOCH']
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        # # Markers loss
        # pred_xyz_67 = (output.pred_xyz_mks_67)[:, :201]
        # target_xyz_67 = labels['target_xyz_67'][:, :pred_xyz_67.shape[1]]
        # target_xyz_67_weight = labels['target_weight_67'][:, :pred_xyz_67.shape[1]]
        # loss_xyz_67 = weighted_l1_loss(pred_xyz_67, target_xyz_67, target_xyz_67_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_loss

        # pred_jts = output.pred_uvd_jts
        sigma = output.sigma
        pred_uvd_jts = output.pred_uvd_jts
        pred_uvd_mks = output.pred_uvd_mks
        pred_uvd = torch.cat([pred_uvd_jts, pred_uvd_mks], dim=1)
        target_uvd_jts = labels['target_uvd_29'].reshape(pred_uvd_jts.shape)
        target_uvd_mks = labels['target_uvd_67'].reshape(pred_uvd_mks.shape)
        target_uvd = torch.cat([target_uvd_jts, target_uvd_mks], dim=1)
        target_uvd_weight_29 = labels['target_weight_29'].reshape(pred_uvd_jts.shape)
        target_uvd_weight_67 = labels['target_weight_67'].reshape(pred_uvd_mks.shape)
        target_uvd_weight = torch.cat([target_uvd_weight_29, target_uvd_weight_67], dim=1)
        nf_loss = nf_loss * target_uvd_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            loss_uvd = nf_loss + Q_logprob
        if self.size_average and target_uvd_weight.sum() > 0:
            # loss_uvd =  loss_uvd.sum() / len(loss_uvd)
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_
            loss_uvd =  loss_uvd.sum() / loss_used_lenth

        else:
            loss_uvd = loss_uvd.sum()

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight
            # loss += loss_xyz_67 * self.xyz67_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_trans=trans_loss,
            loss_scale=scale_loss
        )

        return loss_output


@LOSS.register_module
class L1LossSMPLCamVMRLEFTLoop(nn.Module):
    ''' 
    RLE Regression Loss 3D + SMPL Cam Loss
    '''
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossSMPLCamVMRLEFTLoop, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.ft_theta_weight = self.elements['FT_THETA_WEIGHT']
        self.ft_beta_weight = self.elements['FT_BETA_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = self.elements['PRETRAIN_EPOCH']
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.global_avg_uvd_loss = 0.0
        self.global_loss_count = 0

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def logQ(self, gt_uv, pred_jts, sigma):
        # 1e-9
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        loss_beta_ft = self.criterion_smpl(output.pred_shape_ft * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta_ft = self.criterion_smpl(output.pred_theta_mats_ft * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        # RLE Loss
        nf_loss = output.nf_loss

        # pred_jts = output.pred_uvd_jts
        sigma = output.sigma
        pred_uvd_jts = output.pred_uvd_jts
        pred_uvd_mks = output.pred_uvd_mks
        pred_uvd = torch.cat([pred_uvd_jts, pred_uvd_mks], dim=1)
        target_uvd_jts = labels['target_uvd_29'].reshape(pred_uvd_jts.shape)
        target_uvd_mks = labels['target_uvd_67'].reshape(pred_uvd_mks.shape)
        target_uvd = torch.cat([target_uvd_jts, target_uvd_mks], dim=1)
        target_uvd_weight_29 = labels['target_weight_29'].reshape(pred_uvd_jts.shape)
        target_uvd_weight_67 = labels['target_weight_67'].reshape(pred_uvd_mks.shape)
        target_uvd_weight = torch.cat([target_uvd_weight_29, target_uvd_weight_67], dim=1)
        nf_loss = nf_loss * target_uvd_weight
        residual = True
        if residual:
            Q_logprob = self.logQ(target_uvd, pred_uvd, sigma) * target_uvd_weight
            loss_uvd = nf_loss + Q_logprob
        if self.size_average and target_uvd_weight.sum() > 0:
            # loss_uvd =  loss_uvd.sum() / len(loss_uvd)
            # 找出奇异值
            loss_uvd_weight_ = torch.ones_like(loss_uvd)
            loss_used_lenth = loss_uvd.shape[0]
            if epoch_num > 0:
                # loss_uvd [B, 29, 3]
                for i in range(len(loss_uvd)):
                    loss_sum_bs_ = loss_uvd[i].sum()
                    if loss_sum_bs_ > 0:
                        loss_uvd_weight_[i] = 0.
                        # loss_uvd[i] = 0.
                        loss_used_lenth = loss_used_lenth - 1

            loss_uvd = loss_uvd * loss_uvd_weight_
            loss_uvd =  loss_uvd.sum() / loss_used_lenth

        else:
            loss_uvd = loss_uvd.sum()

        # loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss = loss_twist * self.twist_weight
        

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        loss_ft = loss_beta_ft * self.ft_beta_weight + loss_theta_ft * self.ft_theta_weight
        loss = loss + loss_ft

        loss_output = edict(
            loss=loss,
            loss_uvd=loss_uvd,
            loss_beta=loss_beta,
            loss_theta=loss_theta,
            loss_twist=loss_twist,
            loss_trans=trans_loss,
            loss_scale=scale_loss
        )

        return loss_output


