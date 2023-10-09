"""
Definition of SMPL Parameter Regressor used for regressing the SMPL parameters from the 3D shape
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from .builder import SPPE, build_sppe
# import builder
from litehuman.opt import cfg, logger, opt
# from models.layers import FCBlock, FCResBlock

class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)

class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))
        
    def forward(self, x):
        return F.relu(x + self.fc_block(x))

@SPPE.register_module
class SMPLParamRegressor(nn.Module):
    def __init__(self, use_cpu_svd=True, **kwargs):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        verts_num = kwargs['NUM_VERTS']
        self.layers = nn.Sequential(FCBlock(verts_num * 6, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 10))       # ori: 24*3*3
        self.use_cpu_svd = use_cpu_svd


    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        betas = x.contiguous()
        # rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        # betas = x[:, 24*3*3:].contiguous()
        # rotmat = rotmat.view(-1, 3, 3).contiguous()
        # orig_device = rotmat.device
        # if self.use_cpu_svd:
        #     rotmat = rotmat.cpu()
        # U, S, V = batch_svd(rotmat)

        # rotmat = torch.matmul(U, V.transpose(1,2))
        # det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        # with torch.no_grad():
        #     for i in range(rotmat.shape[0]):
        #         det[i] = torch.det(rotmat[i])
        # rotmat = rotmat * det
        # rotmat = rotmat.view(batch_size, 24, 3, 3)
        # rotmat = rotmat.to(orig_device)
        # rotmat (B, 24, 3, 3) -> mats (B, 24, 4)
    
        output = edict(
            pred_shape=betas
        )
        return output

def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V

def get_regressor(cfg):
    model = build_sppe(cfg)
    if cfg.PRETRAINED:
        logger.info(f'Loading regressor model from {cfg.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.PRETRAINED))
        if cfg.FROZEN:
            for name, parameters in model.named_parameters():
                parameters.requires_grad = False
            logger.info(f'Freezing the parameters of Regressor...')
        else:
            logger.info(f'Start train the parameters of Regressor...')
    else:
        logger.info('Creating the new regressor model...')
    return model
