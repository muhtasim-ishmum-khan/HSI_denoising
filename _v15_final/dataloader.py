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


class MemoryEfficientHSIDataset(Dataset):
    def __init__(self, files, patch_size, noise_level=30,
                 patches_per_file=200, target_bands=None, augment=True, 
                 dataset_type="unknown", train_crop_size=1024, scales=None):
        super().__init__()
        self.files = files
        self.patch_size = patch_size
        self.noise_level = noise_level / 255.0
        self.patches_per_file = patches_per_file
        self.augment = augment
        self.target_bands = target_bands if target_bands else 31  # ICVL default: 31 bands
        self.dataset_type = dataset_type
        self.synthetic_count = 0
        self.real_data_count = 0
        self.train_crop_size = train_crop_size  # ICVL: center crop size
        self.scales = scales if scales else [64, 32, 32]  # ICVL: multi-scale strides

        # Keep existing file loading logic
        self.file_info = []
        print(f"\n=== Initializing {dataset_type.upper()} Dataset ===")
        print(f"Noise Level (σ): {noise_level} -> {self.noise_level:.4f} (0-1 scale)")
        print(f"Center crop size: {train_crop_size}x{train_crop_size}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Multi-scale strides: {self.scales}")

        if len(files) > 0 and files[0] != 'synthetic':
            print(f"Attempting to load {len(files)} files...")
            successful_files = []
            failed_files = []

            for i, file_path in enumerate(files):
                if os.path.exists(file_path):
                    try:
                        print(f"  [{i+1}/{len(files)}] Loading: {os.path.basename(file_path)}")

                        # Try h5py first (for MATLAB v7.3)
                        h5py_success = False
                        try:
                            with h5py.File(file_path, 'r') as f:
                                # FIXED: Look for 'rad' key specifically for ICVL data
                                if 'rad' in f:
                                    key = 'rad'
                                    data = f[key]
                                    cube_shape = data.shape
                                    print(f"    ✓ Shape: {cube_shape}, Key: '{key}' (HDF5)")
                                    self.file_info.append({'path': file_path, 'key': key, 'shape': cube_shape, 'format': 'h5py'})
                                    successful_files.append(os.path.basename(file_path))
                                    h5py_success = True
                                else:
                                    raise Exception("No 'rad' key found in file")
                        except Exception as h5_error:
                            # Fallback to scipy.io for older MAT files
                            if sio and not h5py_success:
                                try:
                                    mat = sio.loadmat(file_path)
                                    key = [k for k in mat.keys() if not k.startswith('__')][0]
                                    cube_shape = mat[key].shape
                                    print(f"    ✓ Shape: {cube_shape}, Key: '{key}' (scipy)")
                                    self.file_info.append({'path': file_path, 'key': key, 'shape': cube_shape, 'format': 'scipy'})
                                    successful_files.append(os.path.basename(file_path))
                                except Exception as scipy_error:
                                    raise Exception(f"Both h5py ({str(h5_error)[:50]}) and scipy ({str(scipy_error)[:50]}) failed")
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
                        failed_files.append(os.path.basename(file_path))

            print(f"\n{dataset_type.upper()} Dataset Summary:")
            print(f"  ✓ Successfully loaded: {len(successful_files)} files")
            print(f"  → Patches per file: {patches_per_file}")
            print(f"  → Total iterations per epoch: {len(successful_files) * patches_per_file}")
            if not self.file_info:
                print(f"  → Will use SYNTHETIC data for {dataset_type}")

    def __len__(self):
        return max(1, len(self.file_info)) * self.patches_per_file

    def _generate_synthetic_data(self):
        """Generate BETTER synthetic HSI data with proper spectral correlation"""
        self.synthetic_count += 1
    
        D, H, W = self.target_bands, self.patch_size, self.patch_size
    
        clean = torch.zeros(D, H, W)
    
        # Create different material signatures
        n_materials = 3
        for mat in range(n_materials):
            signature = torch.randn(D)
            signature = torch.softmax(signature, dim=0)
    
            center_h, center_w = random.randint(H//4, 3*H//4), random.randint(W//4, 3*W//4)
            for h in range(H):
                for w in range(W):
                    dist = ((h - center_h)**2 + (w - center_w)**2) ** 0.5
                    weight = torch.exp(torch.tensor(-dist / (H/4)))  # FIX: Convert to tensor
                    clean[:, h, w] += weight * signature * random.uniform(0.3, 1.0)
    
        # Normalize to [0, 1]
        clean = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)
    
        # Add noise in [0,1] range
        noise = torch.randn_like(clean) * self.noise_level
        noisy = torch.clamp(clean + noise, 0, 1)

        return noisy, clean

    def __getitem__(self, idx):
        # REMOVED: Synthetic data fallback - force real data only
        if not self.file_info:
            raise RuntimeError(
                f"CRITICAL ERROR: No files loaded in {self.dataset_type} dataset!\n"
                f"file_info is empty. Check __init__ file loading."
            )
    
        file_idx = idx // self.patches_per_file
        file_info = self.file_info[file_idx % len(self.file_info)]
    
        # Load based on file format
        if file_info.get('format') == 'h5py':
            with h5py.File(file_info['path'], 'r') as f:
                cube = np.array(f[file_info['key']]).astype(np.float32)
        else:
            mat = sio.loadmat(file_info['path'])
            cube = mat[file_info['key']].astype(np.float32)
        
        # Ensure proper shape (H, W, D)
        if cube.ndim == 3:
            # ICVL data is in (D, H, W) format, need to transpose to (H, W, D)
            if cube.shape[0] < min(cube.shape[1:]):  # Shape is (D, H, W)
                cube = cube.transpose(1, 2, 0)  # Convert to (H, W, D)
            elif cube.shape[2] > min(cube.shape[:2]):  # Shape is likely (H, W, D) already
                pass
            else:
                raise ValueError(
                    f"ERROR: Unexpected cube shape {cube.shape} from file {file_info['path']}\n"
                    f"Cannot determine if format is (D,H,W) or (H,W,D)"
                )
        else:
            raise ValueError(
                f"ERROR: cube has {cube.ndim} dimensions, expected 3\n"
                f"Shape: {cube.shape}, File: {file_info['path']}"
            )
                
        H, W, D = cube.shape
    
        # ICVL: Center crop to train_crop_size (1024x1024)
        if H > self.train_crop_size or W > self.train_crop_size:
            start_h = (H - self.train_crop_size) // 2
            start_w = (W - self.train_crop_size) // 2
            cube = cube[start_h:start_h+self.train_crop_size,
                       start_w:start_w+self.train_crop_size, :]
            H, W = self.train_crop_size, self.train_crop_size
    
        # Check minimum size
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"ERROR: Image too small after cropping\n"
                f"Size: {H}x{W}, Required: {self.patch_size}x{self.patch_size}\n"
                f"File: {file_info['path']}"
            )
        
        if D < 4:
            raise ValueError(
                f"ERROR: Too few spectral bands: {D}\n"
                f"File: {file_info['path']}"
            )
    
        # ICVL: Randomly choose stride from multi-scale strides
        stride = random.choice(self.scales)
        
        # Random patch extraction
        start_h = random.randint(0, H - self.patch_size)
        start_w = random.randint(0, W - self.patch_size)
        cube = cube[start_h:start_h+self.patch_size,
                   start_w:start_w+self.patch_size,
                   :min(self.target_bands, D)]
    
        # ICVL: Normalize to [0, 1]
        cube_min = cube.min()
        cube_max = cube.max()
        if cube_max > cube_min:
            cube = (cube - cube_min) / (cube_max - cube_min)
        else:
            cube = np.clip(cube / (np.max(cube) + 1e-8), 0, 1)
    
        # Convert to (D, H, W) format for model
        cube = cube.transpose(2, 0, 1)
    
        # ============================================================
        # AUGMENTATION PIPELINE (single coherent block)
        # ============================================================
        if self.augment:
            # Geometric augmentations (on numpy arrays)
            # Random rotation (90° increments)
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                cube = np.rot90(cube, k, axes=(1, 2)).copy()
            
            # Horizontal flip
            if random.random() < 0.5:
                cube = np.flip(cube, axis=1).copy()
            
            # Vertical flip
            if random.random() < 0.5:
                cube = np.flip(cube, axis=2).copy()
            
            # Gaussian blur (spatial only) - applied to numpy before tensor conversion
            if random.random() < 0.3:
                sigma = random.uniform(0.3, 0.8)
                for i in range(cube.shape[0]):
                    cube[i] = scipy.ndimage.gaussian_filter(cube[i], sigma=sigma)
    
        # Convert to tensor AFTER all numpy augmentations
        clean_tensor = torch.from_numpy(cube.copy())  # Extra safety: ensure contiguous
    
        # Add noise in [0,1] range
        noise = torch.randn_like(clean_tensor) * self.noise_level
        noisy_tensor = torch.clamp(clean_tensor + noise, 0, 1)
    
        # Spectral augmentation (on tensors, applied to both clean and noisy)
        if self.augment and random.random() < 0.4:
            scale = torch.tensor(
                np.random.uniform(0.95, 1.05, size=(clean_tensor.shape[0], 1, 1)), 
                dtype=torch.float32
            )
            clean_tensor = torch.clamp(clean_tensor * scale, 0, 1)
            noisy_tensor = torch.clamp(noisy_tensor * scale, 0, 1)
    
        self.real_data_count += 1
        return noisy_tensor.float(), clean_tensor.float()

    def get_usage_stats(self):
        """Return statistics about data usage"""
        total = self.real_data_count + self.synthetic_count
        if total == 0:
            return "No data accessed yet"
        real_percent = (self.real_data_count / total) * 100
        synthetic_percent = (self.synthetic_count / total) * 100
        return f"Real data: {self.real_data_count} ({real_percent:.1f}%), Synthetic: {self.synthetic_count} ({synthetic_percent:.1f}%)"
