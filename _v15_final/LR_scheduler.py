import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import json
from skimage.metrics import structural_similarity as compare_ssim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_ema import ExponentialMovingAverage
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
import torch.fft as fft
import h5py
try:
    import scipy.io as sio
except ImportError:
    sio = None
import scipy.ndimage



### COSINE LEARNING RATE SCHEDULER
class WarmupCosineScheduler:
    """
    IMPROVED: LR scheduler with warmup + peak plateau + cosine annealing
    Stays at peak LR longer for better exploration before decay
    """
    def __init__(self, optimizer, warmup_epochs, peak_epochs, total_epochs, lr_max, lr_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs      # Warmup period
        self.peak_epochs = peak_epochs          # NEW: How long to stay at peak
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate and return current LR"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Phase 1: Warmup (linear increase from 0 to lr_max)
            lr = self.lr_max * (self.current_epoch / self.warmup_epochs)
            
        elif self.current_epoch <= self.peak_epochs:
            # Phase 2: Peak plateau (stay at lr_max for exploration)
            lr = self.lr_max
            
        else:
            # Phase 3: Cosine annealing (smooth decay from lr_max to lr_min)
            progress = (self.current_epoch - self.peak_epochs) / (self.total_epochs - self.peak_epochs)
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def state_dict(self):
        """For checkpoint saving compatibility"""
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'peak_epochs': self.peak_epochs,  # Save peak_epochs
            'total_epochs': self.total_epochs,
            'lr_max': self.lr_max,
            'lr_min': self.lr_min
        }
    
    def load_state_dict(self, state_dict):
        """For checkpoint loading compatibility"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.peak_epochs = state_dict.get('peak_epochs', self.warmup_epochs)  # Backward compatibility
        self.total_epochs = state_dict['total_epochs']
        self.lr_max = state_dict['lr_max']
        self.lr_min = state_dict['lr_min']
