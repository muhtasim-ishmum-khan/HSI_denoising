import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import warnings
from einops import rearrange
warnings.filterwarnings('ignore')
import h5py
try:
    import scipy.io as sio
except ImportError:
    sio = None

class LayerNormChannel3d(nn.Module):
    """Lightweight channel normalization"""
    def __init__(self, num_channels: int = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.num_channels = num_channels
        self.gn = None
        if num_channels is not None:
            self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x):
        C = x.shape[1]
        if self.gn is None or C != self.num_channels:
            self.num_channels = C
            self.gn = nn.GroupNorm(1, C, eps=self.eps, affine=True).to(x.device)
        return self.gn(x)

def depthwise_conv3d(channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
    return nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=channels, bias=True, dilation=dilation)

class EfficientChannelAttention(nn.Module):
    """Memory-efficient channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y.expand_as(x)

class AdaptiveDropout3d(nn.Module):
    """Adaptive dropout that adjusts rate based on a factor (e.g., layer depth)"""
    def __init__(self, base_drop=0.25, factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout3d(base_drop * factor)

    def forward(self, x):
        return self.dropout(x)

# ----------------------------
# Memory-Optimized Blocks
# ----------------------------
class PatchMerging3D(nn.Module):
    """
    3D Patch Merging Layer (downsampling)
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Pad if needed
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = x.shape[2:]

        # Convert to (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # Downsample by merging 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, D/2, H/2, W/2, 8*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, D/2, H/2, W/2, 2*C)

        # Convert back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class PatchMerging3D(nn.Module):
    """3D Patch Merging with better handling"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Pad to multiples of 2
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = x.shape[2:]

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class PatchExpanding3D(nn.Module):
    """3D Patch Expanding for decoder"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = self.expand(x)
        
        x = rearrange(x, 'b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c', 
                     p1=2, p2=2, p3=2, c=C//2)
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class SpectralAttentionModule(nn.Module):
    """Dedicated spectral attention for bottleneck - HYBRID ATTENTION"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.norm = nn.GroupNorm(1, dim)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for spectral attention: (B*H*W, num_heads, D, head_dim)
        q = q.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Spectral attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H, W, D, C)
        out = out.permute(0, 4, 3, 1, 2).contiguous()
        
        out = self.proj(out)
        return x + out

# ----------------------------
# SST Blocks
# ----------------------------

class SpectralSelfAttention(nn.Module):
    """
    Spectral Self-Attention: Attends along the spectral dimension (bands)
    Treats each spatial position independently and attends across all bands
    """
    def __init__(self, dim, num_bands, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.dim = dim
            self.num_bands = num_bands  # Expected bands from config
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
            
            # Linear projections for Q, K, V
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            
            # FIXED: Initialize buffer with expected bands, will expand automatically if needed
            spectral_pos_embed = torch.zeros(1, num_bands, dim)
            nn.init.trunc_normal_(spectral_pos_embed, std=0.02)
            self.register_buffer('spectral_pos_embed', spectral_pos_embed)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) - Input feature map
        Returns:
            x: (B, C, D, H, W) - Output feature map
        """
        B, C, D, H, W = x.shape
        
        # Reshape to (B, H, W, D, C) for spectral attention
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, C)
        
        # Flatten spatial dimensions: (B*H*W, D, C)
        x_flat = x.reshape(B * H * W, D, C)
        
        # FIXED: Dynamically expand buffer if we encounter more bands than initialized
        if D > self.spectral_pos_embed.shape[1]:
            # This happens rarely (only when encountering new max bands)
            old_size = self.spectral_pos_embed.shape[1]
            new_size = D
            
            # Create expanded buffer on same device
            expanded = torch.zeros(1, new_size, self.dim, 
                                  device=self.spectral_pos_embed.device,
                                  dtype=self.spectral_pos_embed.dtype)
            
            # Copy existing learned embeddings
            expanded[:, :old_size, :] = self.spectral_pos_embed
            
            # Initialize new bands with small random values
            nn.init.trunc_normal_(expanded[:, old_size:, :], std=0.02)
            
            # Update the buffer in-place
            self.spectral_pos_embed.resize_(expanded.shape)
            self.spectral_pos_embed.copy_(expanded)
            
            #print(f"[SpectralPosEmbed] Auto-expanded from {old_size} to {new_size} bands")
        
        # Use only the bands we need (safe slicing - buffer is always >= D now)
        pos_embed = self.spectral_pos_embed[:, :D, :]  # (1, D, dim)
        
        # Add spectral positional encoding
        x_flat = x_flat + pos_embed
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).reshape(B * H * W, D, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B*H*W, num_heads, D, head_dim)
        
        # Scaled dot-product attention across spectral dimension
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B*H*W, num_heads, D, D)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B * H * W, D, C)  # (B*H*W, D, C)
        
        # Project and reshape back
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # Reshape back to (B, H, W, D, C)
        x_attn = x_attn.reshape(B, H, W, D, C)
        
        # Reshape to original format (B, C, D, H, W)
        x_attn = x_attn.permute(0, 4, 3, 1, 2).contiguous()
        
        return x_attn

class SpatialSelfAttention(nn.Module):
    """
    FIXED Spatial Self-Attention - SAME NAME, IMPROVED IMPLEMENTATION
    Drop-in replacement - no API changes, just better internals
    """
    def __init__(self, dim, num_heads=8, window_size=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # FIXED: Full 3D convolution instead of depthwise
        # OLD: self.dwconv = depthwise_conv3d(dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # NEW: Full conv for cross-spectral information flow
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True)
        
        # Keep same API as before
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # NEW: Relative position bias (optional, improves performance)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x):
        """
        SAME API: (B, C, D, H, W) -> (B, C, D, H, W)
        Just improved internals
        """
        B, C, D, H, W = x.shape
        
        # Apply full 3D conv (cross-spectral enabled)
        x_local = self.dwconv(x)
        
        qkv = self.qkv(x_local)
        
        if H * W > self.window_size ** 2:
            # Efficient attention for large spatial dims
            qkv = rearrange(qkv, 'b (three head c) d h w -> three b d head c (h w)', 
                           three=3, head=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            q_flat = q.reshape(B * D * self.num_heads, self.head_dim, H * W)
            k_flat = k.reshape(B * D * self.num_heads, self.head_dim, H * W)
            v_flat = v.reshape(B * D * self.num_heads, self.head_dim, H * W)
            
            k_global = k_flat.mean(dim=-1, keepdim=True)
            v_global = v_flat.mean(dim=-1, keepdim=True)
            
            q_flat = q_flat * self.scale
            attn = torch.bmm(q_flat.transpose(1, 2), k_global)
            attn = F.softmax(attn, dim=1)
            attn = self.attn_drop(attn)
            
            x_attn = v_global * attn.transpose(1, 2)
            x_attn = x_attn.reshape(B, D, self.num_heads, self.head_dim, H * W)
            
        else:
            # Full attention for small spatial dims
            qkv = rearrange(qkv, 'b (three head c) d h w -> three b d head c (h w)', 
                           three=3, head=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            q_flat = q.reshape(B * D * self.num_heads, self.head_dim, H * W)
            k_flat = k.reshape(B * D * self.num_heads, self.head_dim, H * W)
            v_flat = v.reshape(B * D * self.num_heads, self.head_dim, H * W)
            
            q_flat = q_flat * self.scale
            attn = torch.bmm(q_flat.transpose(1, 2), k_flat)
            
            if H == self.window_size and W == self.window_size:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn.view(B * D, self.num_heads, H * W, H * W) + relative_position_bias.unsqueeze(0)
                attn = attn.view(B * D * self.num_heads, H * W, H * W)
            
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            x_attn = torch.bmm(attn, v_flat.transpose(1, 2))
            x_attn = x_attn.transpose(1, 2)
            x_attn = x_attn.reshape(B, D, self.num_heads, self.head_dim, H * W)
        
        x_attn = rearrange(x_attn, 'b d head c (h w) -> b (head c) d h w', 
                          head=self.num_heads, h=H, w=W)
        
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        return x_attn


class SSTBlock(nn.Module):
    """
    Spectral-Spatial Transformer Block
    Combines spectral and spatial self-attention with feed-forward network
    """
    def __init__(self, dim, num_bands, num_heads=8, window_size=8, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or LayerNormChannel3d
        
        self.dim = dim
        self.num_bands = num_bands
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        # Spectral attention
        self.spectral_attn = SpectralSelfAttention(
            dim=dim,
            num_bands=num_bands,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Spatial attention
        self.spatial_attn = SpatialSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop path for stochastic depth
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
        # Feed-forward network (reuse existing GDFN)
        self.ffn = GDFN(dim, ffn_expansion_factor=mlp_ratio, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            x: (B, C, D, H, W)
        """
        # Spectral attention with residual
        x = x + self.drop_path(self.spectral_attn(self.norm1(x)))
        
        # Spatial attention with residual
        x = x + self.drop_path(self.spatial_attn(self.norm2(x)))
        
        # Feed-forward with residual
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x


class SSTStage(nn.Module):
    """
    SST Stage: Multiple SST blocks with optional downsampling
    Replaces SwinTransformerStage3D
    """
    def __init__(self, dim, num_bands, depth, num_heads=8, window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path_rate=0., norm_layer=None, downsample=None):
        super().__init__()
        norm_layer = norm_layer or LayerNormChannel3d
        
        self.dim = dim
        self.depth = depth
        
        # Stochastic depth decay rule
        dpr = [drop_path_rate * (i / (depth - 1)) if depth > 1 else drop_path_rate 
               for i in range(depth)]
        
        # Build SST blocks
        self.blocks = nn.ModuleList([
            SSTBlock(
                dim=dim,
                num_bands=num_bands,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Downsampling layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            x: (B, C, D, H, W) - features before downsampling
            x_down: (B, 2*C, D/2, H/2, W/2) - downsampled features (if downsample exists)
        """
        # Pass through SST blocks
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        # Downsample if needed
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x

# ----------------------------
# SST Blocks
# ----------------------------

class SpectralSelfModulatingResidualBlock(nn.Module):
    """Spectral Self-Modulating Residual Block (SSMRB) for adaptive feature transformation."""
    def __init__(self, dim, ffn_expand=2, drop=0.25, drop_factor=1.0):
        super().__init__()
        hidden = dim * ffn_expand

        # Main FFN path (vanilla-like)
        self.ffn_pw1 = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=True)
        self.ffn_dw = depthwise_conv3d(hidden * 2, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.ffn_pw2 = nn.Conv3d(hidden, dim, kernel_size=1, bias=True)

        # Self-modulation branch: Generate gamma (scale) and beta (shift) using spectral-adjacent conv
        # Kernel (3,1,1) captures adjacent spectral bands; depthwise for efficiency
        self.mod_gamma = nn.Sequential(
            depthwise_conv3d(dim, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Sigmoid()  # For scaling (0 to 1)
        )
        self.mod_beta = nn.Sequential(
            depthwise_conv3d(dim, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Tanh()  # For shifting (-1 to 1)
        )

        # Normalization and dropout
        self.norm = LayerNormChannel3d(dim)
        self.dropout = AdaptiveDropout3d(drop, factor=drop_factor)  # Adaptive dropout

        # Layer scale for residual contribution
        self.gamma_res = nn.Parameter(torch.ones(1, dim, 1, 1, 1) * 1e-4)

    def _ffn_path(self, x):
        """Vanilla FFN computation."""
        x2 = self.ffn_pw1(x)
        x2 = self.ffn_dw(x2)
        a, b = torch.chunk(x2, 2, dim=1)
        x2 = self.act(a) * b
        x2 = self.ffn_pw2(x2)
        return x2

    def forward(self, x):
        # Normalize input
        x_norm = self.norm(x)

        # Compute main FFN path
        ffn_out = self._ffn_path(x_norm)

        # Compute self-modulation parameters from input (using adjacent spectral info)
        gamma = self.mod_gamma(x_norm)  # Scale
        beta = self.mod_beta(x_norm)    # Shift

        # Apply modulation: gamma * ffn_out + beta
        modulated = gamma * ffn_out + beta

        # Apply dropout and scale
        modulated = self.dropout(modulated) * self.gamma_res

        # Residual connection: x + modulated
        return x + modulated


# ----------------------------
# New Classes Added (for Restormer Integration in Architecture)
# ----------------------------
class LayerNorm3d(nn.Module):
    """3D LayerNorm for channel normalization"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + self.eps)
        out = (x - mu) * inv_std * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return out

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network adapted to 3D, focusing on spatial"""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(hidden * 2, hidden * 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=hidden * 2, bias=bias)  # Spatial focus
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention adapted to 3D, focusing on spatial"""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=dim * 3, bias=bias)  # Spatial focus
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        b, c, d, h, w = q.shape
        q = rearrange(q, 'b (head cc) d h w -> b head cc (d h w)', head=self.num_heads, cc=c // self.num_heads)
        k = rearrange(k, 'b (head cc) d h w -> b head cc (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head cc) d h w -> b head cc (d h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head cc (d h w) -> b (head cc) d h w', head=self.num_heads, d=d, h=h, w=w)
        out = self.project_out(out)
        return out

class RestormerBlock(nn.Module):
    """Restormer Transformer Block adapted to 3D"""
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm3d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
# ----------------------------
# New Classes Added (for Restormer Integration in Architecture)
# ----------------------------

# ----------------------------
# Modified FusedBottleneck (for Restormer Integration in Architecture)
# ----------------------------
# ============================================================
# INSERT THESE NEW CLASSES BEFORE FusedBottleneck (around line ~680)
# These are ADDITIONS, not replacements
# ============================================================

class PositionalEncoding3D(nn.Module):
    """
    FIXED: Lightweight 3D Positional Encoding
    
    Instead of storing full (C, D, H, W) tensor, we use:
    1. Separate 1D embeddings for each dimension (much smaller)
    2. Broadcast and combine at runtime
    
    Memory: O(C*D + C*H + C*W) instead of O(C*D*H*W)
    Example: 64*128 + 64*256 + 64*256 = 41K params vs 537M params!
    """
    def __init__(self, channels, max_d=128, max_h=256, max_w=256):
        super().__init__()
        self.channels = channels
        self.max_d = max_d
        self.max_h = max_h
        self.max_w = max_w
        
        # Separate 1D positional embeddings for each dimension
        # These will be broadcast and combined
        self.pos_embed_d = nn.Parameter(torch.zeros(1, channels, max_d, 1, 1))
        self.pos_embed_h = nn.Parameter(torch.zeros(1, channels, 1, max_h, 1))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, channels, 1, 1, max_w))
        
        # Initialize with small random values
        nn.init.trunc_normal_(self.pos_embed_d, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_h, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_w, std=0.02)
        
        # Learnable scaling factors for each dimension
        self.scale_d = nn.Parameter(torch.ones(1))
        self.scale_h = nn.Parameter(torch.ones(1))
        self.scale_w = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            x + positional encoding: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Slice and broadcast each dimension
        pe_d = self.pos_embed_d[:, :, :D, :, :] * self.scale_d
        pe_h = self.pos_embed_h[:, :, :, :H, :] * self.scale_h
        pe_w = self.pos_embed_w[:, :, :, :, :W] * self.scale_w
        
        # Combine positional encodings (broadcasting happens automatically)
        pe = pe_d + pe_h + pe_w  # Shape: (1, C, D, H, W)
        
        return x + pe


class CrossSpectralSpatialAttention(nn.Module):
    """
    NEW CLASS: Cross-attention between spectral and spatial features
    Enables joint spectral-spatial modeling instead of separate processing
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for cross-attention paths
        self.q_spectral = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_spatial = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_spatial = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_spectral = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_spectral = nn.Linear(dim, dim)
        self.proj_spatial = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """(B, C, D, H, W) -> (B, C, D, H, W)"""
        B, C, D, H, W = x.shape
        
        # Path 1: Spectral features with spatial context
        x_spectral = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        q_spec = self.q_spectral(x_spectral)
        q_spec = q_spec.reshape(B * H * W, D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q_spec_scaled = q_spec * self.scale
        attn_spec = torch.matmul(q_spec_scaled, q_spec_scaled.transpose(-2, -1))
        attn_spec = F.softmax(attn_spec, dim=-1)
        attn_spec = self.attn_drop(attn_spec)
        
        out_spec = torch.matmul(attn_spec, q_spec)
        out_spec = out_spec.transpose(1, 2).reshape(B * H * W, D, C)
        out_spec = self.proj_spectral(out_spec)
        out_spec = self.proj_drop(out_spec)
        out_spectral = out_spec.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        
        # Path 2: Spatial features with spectral context
        x_spatial_q = x.permute(0, 2, 3, 4, 1).reshape(B * D, H * W, C)
        q_spat = self.q_spatial(x_spatial_q)
        q_spat = q_spat.reshape(B * D, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q_spat_scaled = q_spat * self.scale
        attn_spat = torch.matmul(q_spat_scaled, q_spat_scaled.transpose(-2, -1))
        attn_spat = F.softmax(attn_spat, dim=-1)
        attn_spat = self.attn_drop(attn_spat)
        
        out_spat = torch.matmul(attn_spat, q_spat)
        out_spat = out_spat.transpose(1, 2).reshape(B * D, H * W, C)
        out_spat = self.proj_spatial(out_spat)
        out_spat = self.proj_drop(out_spat)
        out_spatial = out_spat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Gated fusion
        concat_features = torch.cat([out_spectral, out_spatial], dim=1)
        gate_input = F.adaptive_avg_pool3d(concat_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        gate = self.gate(gate_input).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        fused = gate * out_spectral + (1 - gate) * out_spatial
        return fused


class EnhancedBottleneck(nn.Module):
    """
    NEW CLASS: SOTA-level bottleneck with cross-attention and joint modeling
    Will be used in the main architecture
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        
        self.norm1 = LayerNormChannel3d(dim)
        self.norm2 = LayerNormChannel3d(dim)
        self.norm3 = LayerNormChannel3d(dim)
        
        # Cross-attention for spectral-spatial joint modeling
        self.cross_attn = CrossSpectralSpatialAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
        # Non-local attention for global context
        self.non_local = nn.Sequential(
            nn.Conv3d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        
        # Keep using existing GDFN and SSMRB
        self.ffn = GDFN(dim, ffn_expansion_factor=mlp_ratio, bias=False)
        self.spectral_refine = SpectralSelfModulatingResidualBlock(
            dim, ffn_expand=2, drop=0.1, drop_factor=1.0
        )
        
        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv3d(dim * 3, dim, 1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """(B, C, D, H, W) -> (B, C, D, H, W)"""
        identity = x
        
        # Path 1: Cross-attention
        cross_out = self.cross_attn(self.norm1(x))
        x = x + cross_out
        
        # Path 2: Non-local
        non_local_weight = self.non_local(self.norm2(x))
        non_local_out = x * non_local_weight
        x = x + non_local_out
        
        # Path 3: FFN + Spectral refinement
        ffn_out = self.ffn(self.norm3(x))
        spectral_out = self.spectral_refine(x + ffn_out)
        
        # Gated fusion
        fusion_input = torch.cat([cross_out, non_local_out, spectral_out], dim=1)
        fusion_weight = self.fusion_gate(fusion_input)
        
        out = identity + fusion_weight * (cross_out + non_local_out + spectral_out) / 3.0
        
        return out

# ----------------------------
# Modified FusedBottleneck (for Restormer Integration in Architecture)
# ----------------------------
        
class FusedBottleneck(nn.Module):
    """
    IMPROVED FusedBottleneck - SAME NAME, better implementation
    Now uses stacked EnhancedBottleneck blocks for SOTA performance
    Drop-in replacement - same API, just calls different internals
    """
    def __init__(self, base_dim, window_sizes=[2, 4]):
        super().__init__()
        # Calculate actual dim from base_dim (maintains compatibility)
        # Your original: dim = base_dim * 4
        # But you call it with base_dim * 2, so actual dim is base_dim * 8
        dim = base_dim * 4  # This gives base_dim * 8 when called with base_dim * 2
        
        # Use stacked enhanced bottleneck blocks instead of old approach
        self.blocks = nn.ModuleList([
            EnhancedBottleneck(dim, num_heads=8, mlp_ratio=4.),
            #EnhancedBottleneck(dim, num_heads=8, mlp_ratio=4.)
        ])
    
    def forward(self, x):
        """
        SAME API: (B, C, D, H, W) -> (B, C, D, H, W)
        """
        for block in self.blocks:
            x = block(x)
        return x

# ----------------------------
# Modified FusedBottleneck (for Restormer Integration in Architecture)
# ----------------------------

# ----------------------------
# Efficient Loss Function
# ----------------------------

# ----------------------------
# Modified MemoryEfficientLoss (for Spatial-Focused Loss Improvements)
# ----------------------------
class MemoryEfficientLoss(nn.Module):
    """Lightweight but effective loss function with FIXED weights and tensor handling"""
    def __init__(self, device='cuda', mse_weight=1.0, l1_weight=1.0, sam_weight=0.5, edge_weight=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.device = device
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target, epoch=None):
        # FIXED: Ensure both tensors have same shape
        if pred.shape != target.shape:
            # If shapes don't match, interpolate pred to match target
            if pred.dim() == 5 and target.dim() == 5:
                pred = F.interpolate(pred, size=target.shape[2:], mode='trilinear', align_corners=False)
            elif pred.dim() == 4 and target.dim() == 4:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        # Main losses
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)

        # FIXED: SAM calculation with proper tensor handling
        eps = 1e-8

        # Handle both 4D and 5D tensors
        if pred.dim() == 5:  # (B, C, D, H, W)
            B, C, D, H, W = pred.shape
            pred_flat = pred.reshape(B, C, D * H * W)  # (B, C, D*H*W)
            target_flat = target.reshape(B, C, D * H * W)  # (B, C, D*H*W)

            # Normalize along channel dimension (spectral bands)
            pred_norm = F.normalize(pred_flat, dim=1, eps=eps)
            target_norm = F.normalize(target_flat, dim=1, eps=eps)

            # Compute cosine similarity along spectral dimension
            cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # (B, D*H*W)

        elif pred.dim() == 4:  # (B, D, H, W) - spectral first
            B, D, H, W = pred.shape
            pred_flat = pred.reshape(B, D, H * W)  # (B, D, H*W)
            target_flat = target.reshape(B, D, H * W)  # (B, D, H*W)

            # Normalize along spectral dimension
            pred_norm = F.normalize(pred_flat, dim=1, eps=eps)
            target_norm = F.normalize(target_flat, dim=1, eps=eps)

            # Compute cosine similarity along spectral dimension
            cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # (B, H*W)

        else:
            # Fallback for other dimensions
            pred_flat = pred.flatten(start_dim=1)
            target_flat = target.flatten(start_dim=1)
            pred_norm = F.normalize(pred_flat, dim=1, eps=eps)
            target_norm = F.normalize(target_flat, dim=1, eps=eps)
            cos_sim = torch.sum(pred_norm * target_norm, dim=1)

        cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)
        sam_loss = torch.mean(1 - cos_sim)

        # FIXED: Edge loss with proper spatial dimension handling
        def spatial_gradient(x):
            if x.dim() == 5:  # (B, C, D, H, W)
                grad_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
                grad_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
            elif x.dim() == 4:  # (B, D, H, W)
                grad_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
                grad_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            else:
                return 0, 0
            return grad_h.mean(), grad_w.mean()

        pred_grad_h, pred_grad_w = spatial_gradient(pred)
        target_grad_h, target_grad_w = spatial_gradient(target)
        edge_loss = abs(pred_grad_h - target_grad_h) + abs(pred_grad_w - target_grad_w)

        # Static combination
        total_loss = (
            self.mse_weight * mse_loss +
            self.l1_weight * l1_loss +
            self.sam_weight * sam_loss +
            self.edge_weight * edge_loss
        )

        return total_loss
# ----------------------------
# Modified MemoryEfficientLoss (for Spatial-Focused Loss Improvements)
# ----------------------------

# ----------------------------
# Memory-Efficient U-Net
# ----------------------------


# ----------------------------
# Memory-Efficient U-Net
# ----------------------------
class MemoryOptimizedUNet(nn.Module):
    """
    SST-based U-Net for HSI Denoising (MODIFIED)
    - 4 hierarchical stages with SST blocks instead of Swin
    - Spectral-aware attention at all levels
    - Deep supervision at each decoder stage
    """
    def __init__(self, in_channels=1, base_dim=48, window_sizes=[4, 8, 16], num_bands=64):
        super().__init__()
        self.base_dim = base_dim
        self.in_channels = in_channels
        self.num_bands = num_bands
        
        # Initial projection
        self.patch_embed = nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1)
        self.pos_embed_init = PositionalEncoding3D(base_dim, 128, 256, 256)
        
        # ENCODER: 4 SST stages with [2, 2, 6, 2] depth
        # Stage 1: base_dim, shallow features
        self.enc_stage1 = SSTStage(
            dim=base_dim,
            num_bands=num_bands,
            depth=2,
            num_heads=8,
            window_size=8,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.05,
            downsample=PatchMerging3D
        )
        
        # Stage 2: base_dim*2, intermediate features
        self.enc_stage2 = SSTStage(
            dim=base_dim * 2,
            num_bands=num_bands // 2,  # Bands halved after merging
            depth=2,
            num_heads=8,
            window_size=8,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.1,
            downsample=PatchMerging3D
        )
        
        # Stage 3: base_dim*4, deep features (6 blocks)
        self.enc_stage3 = SSTStage(
            dim=base_dim * 4,
            num_bands=num_bands // 4,
            depth=6,
            num_heads=16,
            window_size=4,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.15,
            downsample=PatchMerging3D
        )
        
        # Stage 4 (deepest): base_dim*8, bottleneck
        self.enc_stage4 = SSTStage(
            dim=base_dim * 8,
            num_bands=num_bands // 8,
            depth=2,
            num_heads=16,
            window_size=2,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.2,
            downsample=None
        )

        self.pe_enc1 = PositionalEncoding3D(base_dim, 128, 256, 256)
        self.pe_enc2 = PositionalEncoding3D(base_dim * 2, 64, 128, 128)
        self.pe_enc3 = PositionalEncoding3D(base_dim * 4, 32, 64, 64)
        self.pe_enc4 = PositionalEncoding3D(base_dim * 8, 16, 32, 32)
        
        # BOTTLENECK: Keep your custom FusedBottleneck (works well with SST)
        self.spectral_attention = SpectralAttentionModule(base_dim * 8, num_heads=8)
        self.bottleneck_fusion = FusedBottleneck(base_dim * 2, window_sizes=window_sizes)
        
        # DECODER: 4 SST stages matching encoder
        # Stage 3 decoder
        self.dec_stage3_up = PatchExpanding3D(dim=base_dim * 8)
        self.dec_stage3 = SSTStage(
            dim=base_dim * 4,
            num_bands=num_bands // 4,
            depth=6,
            num_heads=16,
            window_size=4,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.1,
            downsample=None
        )
        
        # Stage 2 decoder
        self.dec_stage2_up = PatchExpanding3D(dim=base_dim * 4)
        self.dec_stage2 = SSTStage(
            dim=base_dim * 2,
            num_bands=num_bands // 2,
            depth=2,
            num_heads=8,
            window_size=8,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.08,
            downsample=None
        )
        
        # Stage 1 decoder
        self.dec_stage1_up = PatchExpanding3D(dim=base_dim * 2)
        self.dec_stage1 = SSTStage(
            dim=base_dim,
            num_bands=num_bands,
            depth=2,
            num_heads=8,
            window_size=8,
            mlp_ratio=4.,
            drop=0.0,
            attn_drop=0.0,
            drop_path_rate=0.02,
            downsample=None
        )
        
        # DEEP SUPERVISION: Auxiliary outputs at each decoder stage
        self.deep_sup3 = nn.Conv3d(base_dim * 4, in_channels, 1)
        self.deep_sup2 = nn.Conv3d(base_dim * 2, in_channels, 1)
        self.deep_sup1 = nn.Conv3d(base_dim, in_channels, 1)
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv3d(base_dim, base_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_dim // 2, in_channels, 1),
        )
        
        # Global residual
        self.global_residual = nn.Conv3d(in_channels, in_channels, 1)
        
        # Deep supervision flag
        self.use_deep_supervision = True

    def _align_tensors(self, x, target_size):
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x

    def forward(self, x, return_deep_sup=False):
        # Handle input shape
        original_was_4d = False
        if x.dim() == 4:
            original_was_4d = True
            x = x.unsqueeze(1)
        elif x.dim() == 5 and x.shape[1] != 1:
            if x.shape[2] == 1:
                x = x.transpose(1, 2)
        
        original_size = x.shape[2:]
        input_residual = self.global_residual(x)
        
        # Initial embedding
        x = self.patch_embed(x)
        x = self.pos_embed_init(x)
        
        # ENCODER (4 SST stages)
        e1, e1_down = self.enc_stage1(self.pe_enc1(x))       # Skip 1
        e2, e2_down = self.enc_stage2(self.pe_enc2(e1_down))  # Skip 2
        e3, e3_down = self.enc_stage3(self.pe_enc3(e2_down))  # Skip 3
        e4, _ = self.enc_stage4(self.pe_enc4(e3_down))        # Deepest features
        
        # BOTTLENECK: Hybrid attention
        b = self.spectral_attention(e4)  # Add spectral attention
        b = self.bottleneck_fusion(b)   # Your custom fusion
        
        # DECODER with deep supervision
        deep_outputs = []
        
        # Decoder stage 3
        d3 = self.dec_stage3_up(b)
        d3 = self._align_tensors(d3, e3.shape[2:])
        d3 = d3 + e3  # Skip connection
        d3, _ = self.dec_stage3(d3)
        if self.training and self.use_deep_supervision:
            sup3 = self.deep_sup3(d3)
            sup3 = self._align_tensors(sup3, original_size)
            deep_outputs.append(sup3)
        
        # Decoder stage 2
        d2 = self.dec_stage2_up(d3)
        d2 = self._align_tensors(d2, e2.shape[2:])
        d2 = d2 + e2
        d2, _ = self.dec_stage2(d2)
        if self.training and self.use_deep_supervision:
            sup2 = self.deep_sup2(d2)
            sup2 = self._align_tensors(sup2, original_size)
            deep_outputs.append(sup2)
        
        # Decoder stage 1
        d1 = self.dec_stage1_up(d2)
        d1 = self._align_tensors(d1, e1.shape[2:])
        d1 = d1 + e1
        d1, _ = self.dec_stage1(d1)
        if self.training and self.use_deep_supervision:
            sup1 = self.deep_sup1(d1)
            sup1 = self._align_tensors(sup1, original_size)
            deep_outputs.append(sup1)
        
        # Final reconstruction
        out = self.final_conv(d1)
        out = self._align_tensors(out, original_size)
        input_residual = self._align_tensors(input_residual, original_size)
        out = out + input_residual
        
        # Return format handling
        if original_was_4d and out.shape[1] == 1:
            out = out.squeeze(1)
            if self.training and self.use_deep_supervision:
                deep_outputs = [o.squeeze(1) for o in deep_outputs]
        
        if return_deep_sup and self.training:
            return out, deep_outputs
        return out