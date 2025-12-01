import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import glob
from skimage.metrics import structural_similarity as compare_ssim
import h5py
try:
    import scipy.io as sio
except ImportError:
    sio = None
import scipy.ndimage


class metrics_hsi(nn.Module):
    def calculate_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2)
        return 10 * torch.log10(1.0 / (mse + 1e-8))
    
    def calculate_ssim(pred_np, target_np):
        """Calculate SSIM across all spectral bands"""
        if pred_np.ndim == 3:  # (D, H, W)
            D, H, W = pred_np.shape
            ssim_vals = []
            for d in range(D):
                try:
                    ssim_val = compare_ssim(pred_np[d], target_np[d], data_range=1.0)
                    ssim_vals.append(ssim_val)
                except Exception:
                    ssim_vals.append(0.5)  # Fallback
            return np.mean(ssim_vals)
        else:
            return compare_ssim(pred_np, target_np, data_range=1.0)
    
    def calculate_sam(pred_np, target_np):
        """Calculate Spectral Angle Mapper"""
        eps = 1e-8
        if pred_np.ndim == 3:  # (D, H, W)
            pred_flat = pred_np.reshape(pred_np.shape[0], -1)  # (D, H*W)
            target_flat = target_np.reshape(target_np.shape[0], -1)
    
            dot = np.sum(pred_flat * target_flat, axis=0)
            norm_pred = np.linalg.norm(pred_flat, axis=0) + eps
            norm_target = np.linalg.norm(target_flat, axis=0) + eps
            cos_angle = np.clip(dot / (norm_pred * norm_target), -1, 1)
            angles = np.arccos(cos_angle)
            return np.mean(angles)
        else:
            dot = np.sum(pred_np * target_np)
            norm_pred = np.linalg.norm(pred_np) + eps
            norm_target = np.linalg.norm(target_np) + eps
            cos_angle = np.clip(dot / (norm_pred * norm_target), -1, 1)
            return np.arccos(cos_angle)