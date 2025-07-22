# FILE TO CREATE CUSTOM MODEL FOR ARCHITECTURE 2::
# IMPORTS::

import torch
import torch.nn as nn
import torch.nn.functional as F


# UFORMER BASIC MODEL 


class WindowAttention(nn.Module):
    def init(self, dim, num_heads):
        super().init()
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, batch_first = True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1,2)
        attn_output, _ = self.attn(x, x, x)
        return attn_output.transpose(1,2).view(B,C,H,W)


class TransformerBlock(nn.Module):
    def init(self, dim, num_heads):
        super().init()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, -1).permute(0,2,1)
        x = x + self.attn(self.norm1(x_reshaped)).permute(0,2,1).view(B, C, H, W)
        x_reshaped = x.view(B, C, -1).permute(0,2,1)
        x = x + self.ff(self.norm2(x_reshaped)).permute(0,2,1).view(B, C, H, W)
        return x

class Uformer(nn.Module):
    def init(self, in_channels, embed_dim = 64, heads = 4):
        super().init()
        self.encoder1 = nn.Conv2d(in_channels, embed_dim, kernel_size = 3, padding = 1)
        self.transformer1 = TransformerBlock(embed_dim, heads)
        self.down1 = nn.Conv2d(embed_dim, embed_dim*2 , kernel_size = 4, stride = 2, padding = 1)

        self.transformer2 = TransformerBlock(embed_dim * 2, heads)

        self.up1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size = 4, stride = 2, padding = 1)
        self.decoder1 = nn.Conv2d(embed_dim, in_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.transformer1(x1)

        x2 = self.down1(x1)
        x2 = self.transformer2(x2)

        x3 = self.up1(x2) + x1
        out = self.decoder1(x3)
        return out
 
        