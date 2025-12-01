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


def create_training_visualizations(train_losses, val_losses, val_psnrs, val_ssims, val_sams, learning_rates, save_dir, best_psnr):
    """Create comprehensive training visualizations and save them"""
    import matplotlib.pyplot as plt
    plt.style.use('default')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'HSI Denoising Training Progress (Best PSNR: {best_psnr:.4f} dB)', fontsize=16)

    # Calculate validation epochs
    val_epochs = [i*5 for i in range(1, len(val_losses)+1)] if val_losses else []

    # 1. Training and Validation Loss
    axes[0, 0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss', alpha=0.8)
    if val_losses:
        axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', marker='o', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. PSNR over Epochs
    if val_psnrs:
        axes[0, 1].plot(val_epochs, val_psnrs, 'r-', label='PSNR', marker='o', markersize=4)
        axes[0, 1].axhline(y=40, color='purple', linestyle='--', label='Target (40 dB)', alpha=0.7)
        axes[0, 1].axhline(y=best_psnr, color='orange', linestyle=':', label=f'Best ({best_psnr:.2f} dB)', alpha=0.7)
        axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR over Epochs')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. SSIM over Epochs
    if val_ssims:
        axes[0, 2].plot(val_epochs, val_ssims, 'r-', label='SSIM', marker='o', markersize=4)
        axes[0, 2].axhline(y=1.0, color='purple', linestyle='--', label='Perfect (1.0)', alpha=0.7)
        axes[0, 2].legend()
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM')
    axes[0, 2].set_title('SSIM over Epochs')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.1])

    # 4. SAM over Epochs
    if val_sams:
        axes[1, 0].plot(val_epochs, val_sams, 'r-', label='SAM', marker='o', markersize=4)
        axes[1, 0].axhline(y=0, color='purple', linestyle='--', label='Perfect (0.0)', alpha=0.7)
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SAM (radians)')
    axes[1, 0].set_title('SAM over Epochs')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Learning Rate Changes
    axes[1, 1].plot(range(1, len(learning_rates)+1), learning_rates, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    # 6. Combined Metrics Summary
    if val_psnrs and val_ssims and val_sams:
        # Normalize metrics for combined view
        norm_psnr = np.array(val_psnrs) / 50.0  # Normalize to ~1
        norm_ssim = np.array(val_ssims)  # Already 0-1
        norm_sam = 1 - np.array(val_sams)  # Invert so higher is better

        axes[1, 2].plot(val_epochs, norm_psnr, 'b-', label='PSNR (normalized)', alpha=0.7, linewidth=2)
        axes[1, 2].plot(val_epochs, norm_ssim, 'r-', label='SSIM', alpha=0.7, linewidth=2)
        axes[1, 2].plot(val_epochs, norm_sam, 'g-', label='1-SAM', alpha=0.7, linewidth=2)
        axes[1, 2].legend()

    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Normalized Metrics')
    axes[1, 2].set_title('Combined Metrics Overview')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1.1])

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training plots saved to: {plot_path}")

    # Also save as PDF for high quality
    plot_path_pdf = os.path.join(save_dir, 'training_progress.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight', facecolor='white')

    plt.show()
    plt.close()

def print_final_scores(val_psnrs, val_ssims, val_sams, best_psnr, train_files, val_files, save_dir):
    """Print comprehensive final scores"""
    print("\n" + "="*80)
    print(f"Best Validation PSNR: {best_psnr:.4f} dB")
    print("Training finished. Models saved with comprehensive metadata.")
    print(f"Results saved to: {save_dir}")

    print("\n=== FINAL SCORES ===")
    if val_psnrs and val_ssims and val_sams:
        # Handle single validation set
        final_psnr = val_psnrs[-1]
        final_ssim = val_ssims[-1]
        final_sam = val_sams[-1]

        print(f"Final Validation PSNR: {final_psnr:.4f} dB")
        print(f"Final Validation SSIM: {final_ssim:.4f}")
        print(f"Final Validation SAM: {final_sam:.4f} radians")
        
        # Additional statistics
        print(f"\nValidation History:")
        print(f"  Mean PSNR: {np.mean(val_psnrs):.4f} ± {np.std(val_psnrs):.4f} dB")
        print(f"  Mean SSIM: {np.mean(val_ssims):.4f} ± {np.std(val_ssims):.4f}")
        print(f"  Mean SAM:  {np.mean(val_sams):.4f} ± {np.std(val_sams):.4f} radians")
        print(f"  Best PSNR: {max(val_psnrs):.4f} dB")
        print(f"  Worst PSNR: {min(val_psnrs):.4f} dB")

    # Count actual files (filter out 'synthetic')
    real_train_files = [f for f in train_files if f != 'synthetic']
    real_val_files = [f for f in val_files if f != 'synthetic']

    print(f"\nDataset Information:")
    print(f"  Training files: {len(real_train_files)}")
    print(f"  Validation files: {len(real_val_files)}")
    print("="*80)
