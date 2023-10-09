import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SMPLBetaRegressor(nn.Module):
    def __init__(self):
        super(SMPLBetaRegressor, self).__init__()
        self.layers = nn.Sequential(
            FCBlock(54*3, 1024),
            FCResBlock(1024, 1024),
            FCResBlock(1024, 1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        """
        Forward pass.
        Input:
            x: size = (B, 54 * 3)
        Returns:
            SMPL shape params: size = (B, 10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        betas = self.layers(x)
        return betas