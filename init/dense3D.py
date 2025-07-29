import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import os
import gc
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import random

class ResidualConv3DProgressiveRefinement(nn.Module):
    """
    Feed Forward Block: Residual Conv3D with Progressive Refinement for HSI Denoising
    Takes aggregated Conv3D + GSSA features and performs progressive denoising
    """
    def __init__(self, in_channels=64, out_channels=1, dropout=0.1, use_groupnorm=True):
        super(ResidualConv3DProgressiveRefinement, self).__init__()

        self.use_groupnorm = use_groupnorm

        # Feature aggregation layer - learnable weight for Conv3D + GSSA combination
        self.aggregation_weight = nn.Parameter(torch.tensor(0.5))  # Learnable α

        # Progressive refinement blocks
        # Block 1: General spatial-spectral features (3x3x3)
        self.block1_conv = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=False)
        self.block1_norm = self._get_norm_layer(in_channels//2)
        self.block1_residual = nn.Conv3d(in_channels, in_channels//2, kernel_size=1, bias=False)

        # Block 2: Enhanced spectral correlation (3x3x5) - leverages GSSA's spectral selection
        self.block2_conv = nn.Conv3d(in_channels//2, in_channels//4, kernel_size=(3, 3, 5), padding=(1, 1, 2), bias=False)
        self.block2_norm = self._get_norm_layer(in_channels//4)
        self.block2_residual = nn.Conv3d(in_channels//2, in_channels//4, kernel_size=1, bias=False)

        # Block 3: Spectral refinement (1x1x3) - pure spectral processing
        self.block3_conv = nn.Conv3d(in_channels//4, in_channels//8, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        self.block3_norm = self._get_norm_layer(in_channels//8)
        self.block3_residual = nn.Conv3d(in_channels//4, in_channels//8, kernel_size=1, bias=False)

        # Final denoising layer - reconstructs clean patches
        self.final_conv = nn.Conv3d(in_channels//8, out_channels, kernel_size=3, padding=1)

        # Dropout for regularization
        self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights
        self._initialize_weights()

    def _get_norm_layer(self, channels):
        if self.use_groupnorm:
            # Use groups of 8, fallback to channels if less than 8
            groups = min(8, channels)
            return nn.GroupNorm(groups, channels)
        else:
            return nn.BatchNorm3d(channels)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, conv3d_features, gssa_features):
        """
        Args:
            conv3d_features: (B, C, D, H, W) features from Conv3D
            gssa_features: (B, C, D, H, W) features from GSSA (meaningful band selection)
        Returns:
            denoised: (B, 1, D, H, W) denoised patches
        """
        # Feature aggregation with learnable weight
        # conv3d provides local features, gssa provides global spectral context
        aggregated = conv3d_features + self.aggregation_weight * gssa_features

        # Block 1: General spatial-spectral features (3x3x3)
        x1 = F.gelu(self.block1_norm(self.block1_conv(aggregated)))
        x1 = x1 + self.block1_residual(aggregated)  # Residual connection
        x1 = self.dropout(x1)

        # Block 2: Enhanced spectral correlation (3x3x5)
        # Larger spectral kernel leverages GSSA's meaningful band selection
        x2 = F.gelu(self.block2_norm(self.block2_conv(x1)))
        x2 = x2 + self.block2_residual(x1)  # Residual connection
        x2 = self.dropout(x2)

        # Block 3: Pure spectral refinement (1x1x3)
        # Focus on spectral consistency without spatial mixing
        x3 = F.gelu(self.block3_norm(self.block3_conv(x2)))
        x3 = x3 + self.block3_residual(x2)  # Residual connection
        x3 = self.dropout(x3)

        # Final denoising - reconstruct clean patches
        denoised = self.final_conv(x3)

        return denoised