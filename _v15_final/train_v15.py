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
import multiprocessing
import time
import sys
import h5py
try:
    import scipy.io as sio
except ImportError:
    sio = None
import scipy.ndimage

from metrics_hsi import metrics_hsi
from dataloader import MemoryEfficientHSIDataset
from model_v15 import MemoryOptimizedUNet, MemoryEfficientLoss
from visualization import print_final_scores, create_training_visualizations
from LR_scheduler import WarmupCosineScheduler


### MAIN FUNCTION TO RUN EPOCH-WISE TRAINING OF THE ADAPTIVE SPATIAL-SPECTRAL MODEL
def main():
    print("=== SOTA-Enhanced Memory-Optimized HSI Denoising ===")
    print("4-stage SST Transformer with hybrid attention and deep supervision")
    print("Target: PSNR > 40 dB with efficient memory usage")

    # Enable TF32 for RTX 40-series (faster matrix operations)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # UPDATED CONFIG for 4-stage
    config = {
        'patch_size': 64,           
        'batch_size': 2,             
        'base_dim': 64,              
        'noise_level': 30,           
        'lr_max': 1.5e-4,          
        'lr_min': 1e-6,              
        'total_epochs': 300,         
        'patience': 20,             
        'weight_decay': 2e-4,       
        'val_split': 0.1,            
        'seed': 42,                  
        'target_bands': 31,
        'gradient_accumulation': 8,   # Simulate batch_size= batch_size * gradient_accumulation
        'train_crop_size': 1024,
        'scales': [64, 32, 32], 
        'warmup_epochs': 10,
    }

    # Set seeds
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    # else:
    #     device = 'cpu'
    # print(f"Device: {device}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = '/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/test_exp/icvl_part/train'
    save_dir = './HSI_denoising_ICVL_resultsV15_noise30'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== DATA DISCOVERY ===")
    print(f"Looking for HSI data in: {data_dir}")

    # Dataset preparation with detailed logging
    if os.path.exists(data_dir):
        try:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            print(f"Found {len(mat_files)} .mat files in directory")

            if len(mat_files) >= 10:
                n_val = 10
                train_files = mat_files[:-n_val]  # First 90 for training
                val_files = mat_files[-n_val:]     # Last 10 for validation
                print(f"ICVL Split: {len(train_files)} training files, {len(val_files)} validation files")
            else:
                print(f"Warning: Found only {len(mat_files)} files. Need at least 10 for ICVL setup.")
                print("Using what's available with proportional split...")
                n_val = max(1, len(mat_files) // 10)
                train_files = mat_files[:-n_val]
                val_files = mat_files[-n_val:]
        except Exception as e:
            print(f"Error accessing data directory: {e}")
            train_files, val_files = ['synthetic'], ['synthetic']
    else:
        print(f"Data directory not found: {data_dir}")
        train_files, val_files = ['synthetic'], ['synthetic']

    # Create datasets with enhanced logging
    train_dataset = MemoryEfficientHSIDataset(
        train_files,
        patch_size=config['patch_size'],
        noise_level=config['noise_level'],
        patches_per_file=15,
        target_bands=config['target_bands'],
        augment=True,
        dataset_type="training",
        train_crop_size=config['train_crop_size'],
        scales=config['scales']
    )

    val_dataset_synthetic = MemoryEfficientHSIDataset(
        val_files,
        patch_size=config['patch_size'],
        noise_level=config['noise_level'],
        patches_per_file=10,
        target_bands=config['target_bands'],
        augment=False,
        dataset_type="validation_synthetic",
        train_crop_size=config['train_crop_size'],
        scales=[config['patch_size']]  # No multi-scale for validation
    )


    # multiprocessing.set_start_method('spawn', force=True) ### for mac m1 support of multiplle workers
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader_synthetic = DataLoader(val_dataset_synthetic, batch_size=1,
                           shuffle=False, num_workers=2, pin_memory=True)

    # Pre-training data verification
    print("\n=== PRE-TRAINING DATA VERIFICATION ===")
    print("Testing first few training samples...")

    test_count = 3
    samples_verified = 0
    
    for i, (noisy, clean) in enumerate(train_loader):
        if i >= test_count:
            break
        print(f"Sample {i+1}: Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
        samples_verified += noisy.shape[0]
    
    if train_dataset.file_info:
        print(f"✓ Successfully verified {samples_verified} samples from {len(train_dataset.file_info)} real data files")
    else:
        print(f"⚠ Using synthetic data (no real files loaded)")
    
    print("=" * 50)

    # Model setup - NEW: SOTA 4-stage Swin Transformer
    model = MemoryOptimizedUNet(
        in_channels=1,
        base_dim=config['base_dim'],
        window_sizes=[4,8,16],
        num_bands=config['target_bands']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    #print(f"Architecture: 4-stage SST Transformer + Hybrid Attention + Deep Supervision")
    print(f"Architecture: 4-stage SST Transformer + Hybrid Attention")

    # Initialize EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    print("EMA initialized with decay=0.999")

    # Loss and optimizer
    criterion = MemoryEfficientLoss(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr_max'],
                            betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config['weight_decay'])

    # Scheduler
    #from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=config['lr_min'])
    #scheduler = CosineAnnealingLR(optimizer, T_max=config['total_epochs'], eta_min=config['lr_min'])
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config['warmup_epochs'],
        peak_epochs=50,
        total_epochs=config['total_epochs'],
        lr_max=config['lr_max'],
        lr_min=config['lr_min']
    )
    print(f"Scheduler: Warmup ({config['warmup_epochs']} epochs) + Cosine Annealing")
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Training tracking
    best_psnr = 0
    patience_counter = 0
    train_losses, val_losses, val_psnrs, val_ssims, val_sams = [], [], [], [], []
    learning_rates = []

    # NEW: Gradient accumulation setup
    accumulation_steps = config.get('gradient_accumulation', 1)
    print(f"Using gradient accumulation: {accumulation_steps} steps (effective batch size: {config['batch_size'] * accumulation_steps})")

    total_iterations = len(train_loader)
    print(f"Total iterations per epoch: {total_iterations}")

    print(f"Starting training for {config['total_epochs']} epochs...")
    print("-" * 80)

    for epoch in range(1, config['total_epochs'] + 1):
        # ============================================================
        # TRAINING PHASE - WITH DEEP SUPERVISION & GRADIENT ACCUMULATION
        # ============================================================
        model.train()
        epoch_loss = 0
        epoch_main_loss = 0
        epoch_deep_loss = 0
        num_batches = 0

        optimizer.zero_grad()  # Initialize outside loop

        epoch_start_time = time.time()
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.float().to(device, non_blocking=True)
            clean = clean.float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                # NEW: Get main output and deep supervision outputs
                output, deep_outputs = model(noisy, return_deep_sup=True)
                
                # Main loss
                main_loss = criterion(output, clean)
                
                # NEW: Deep supervision losses (weighted progressively)
                deep_loss = 0
                if len(deep_outputs) > 0:
                    weights = [0.4, 0.3, 0.3]  # Weights for 3 auxiliary outputs
                    for i, deep_out in enumerate(deep_outputs):
                        deep_loss += weights[i] * criterion(deep_out, clean)
                
                # Combined loss with gradient accumulation scaling
                #deep supervision loss disabled
                #loss = (main_loss + 0.3 * deep_loss) / accumulation_steps
                loss = main_loss / accumulation_steps

            scaler.scale(loss).backward()

            # NEW: Step optimizer only every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                #EMA
                ema.update()

            # Logging (unscale loss for accurate reporting)
            epoch_loss += loss.item() * accumulation_steps
            epoch_main_loss += main_loss.item()
            epoch_deep_loss += deep_loss.item() if isinstance(deep_loss, torch.Tensor) else deep_loss
            num_batches += 1

            # Memory management
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()


            # NEW: Real-time dynamic progress display
            elapsed_time = time.time() - epoch_start_time
            progress_pct = ((batch_idx + 1) / total_iterations) * 100
            
            # Estimate time remaining
            if batch_idx > 0:
                time_per_iter = elapsed_time / (batch_idx + 1)
                remaining_iters = total_iterations - (batch_idx + 1)
                eta_seconds = time_per_iter * remaining_iters
                eta_mins = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_mins}m {eta_secs}" if eta_mins > 0 else f"{eta_secs}s"
            
            # Dynamic progress bar (overwrites same line)
            bar_length = 30
            filled_length = int(bar_length * (batch_idx + 1) // total_iterations)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r  Epoch {epoch} [{batch_idx + 1}/{total_iterations}] {bar} {progress_pct:.1f}%')
            sys.stdout.flush()

        # NEW: Print newline after progress bar completes
        print()  # Move to next line after epoch completes

        # NEW: Calculate total epoch time
        epoch_time = time.time() - epoch_s



        current_lr = scheduler.step()
        learning_rates.append(current_lr)
        train_loss = epoch_loss / num_batches
        train_main_loss = epoch_main_loss / num_batches
        train_deep_loss = epoch_deep_loss / num_batches
        train_losses.append(train_loss)

        # Data usage statistics every 50 epochs
        #if epoch % 50 == 0:
            #print(f"  Data usage after {epoch} epochs: {train_dataset.get_usage_stats()}")
            #print(f"  Loss breakdown - Main: {train_main_loss:.6f}, Deep: {train_deep_loss:.6f}")

        # ============================================================
        # VALIDATION PHASE - WITHOUT DEEP SUPERVISION
        # ============================================================
        if epoch % 5 == 0:
            model.eval()

            #EMA
            with ema.average_parameters():
                val_loss = 0
                total_psnr = 0
                total_ssim = 0
                total_sam = 0
                num_val_batches = 0
            
                with torch.no_grad():
                    for noisy, clean in val_loader_synthetic:
                        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            
                        with torch.amp.autocast('cuda'):
                            output = model(noisy, return_deep_sup=False)
                            loss = criterion(output, clean)
            
                        val_loss += loss.item()
                        psnr = calculate_psnr(output, clean)
                        total_psnr += psnr.item()
            
                        output_np = output.squeeze(0).cpu().numpy() if output.dim() == 4 else output.squeeze(0).squeeze(0).cpu().numpy()
                        clean_np = clean.squeeze(0).cpu().numpy() if clean.dim() == 4 else clean.squeeze(0).squeeze(0).cpu().numpy()
            
                        ssim_val = calculate_ssim(output_np, clean_np)
                        sam_val = calculate_sam(output_np, clean_np)
            
                        total_ssim += ssim_val
                        total_sam += sam_val
                        num_val_batches += 1
            
                val_loss /= num_val_batches
                val_psnr = total_psnr / num_val_batches
                val_ssim = total_ssim / num_val_batches
                val_sam = total_sam / num_val_batches
            
                # **FIX: Append metrics to tracking lists**
                val_losses.append(val_loss)
                val_psnrs.append(val_psnr)
                val_ssims.append(val_ssim)
                val_sams.append(val_sam)
            
                # Check improvement
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        #EMA
                        'ema_state_dict': ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_psnr': best_psnr,
                        'config': config
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"  *** New best model saved! PSNR: {best_psnr:.4f} dB ***")
                else:
                    patience_counter += 1
            
                print(f"Epoch {epoch:4d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.4f} | "
                      f"Val SSIM: {val_ssim:.4f} | Val SAM: {val_sam:.4f} | Best: {best_psnr:.4f}")
            
            # Early stopping
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}. Best PSNR: {best_psnr:.4f} dB")
                break
        else:
            print(f"Epoch {epoch:4d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f}")

        # Clear memory
        torch.cuda.empty_cache()


    # Save complete model with all metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_psnr': best_psnr,
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'val_sams': val_sams,
        'learning_rates': learning_rates,
        'architecture': '4-stage SST Transformer + Hybrid Attention + Deep Supervision',
        'total_params': total_params
    }, os.path.join(save_dir, 'enhanced_denoising_pipeline_full.pth'))

    # Create comprehensive visualizations
    print("\nCreating training visualizations...")
    create_training_visualizations(
        train_losses=train_losses,
        val_losses=val_losses,
        val_psnrs=val_psnrs,
        val_ssims=val_ssims,
        val_sams=val_sams,
        learning_rates=learning_rates,
        save_dir=save_dir,
        best_psnr=best_psnr
    )

    # Print final comprehensive scores
    print_final_scores(
        val_psnrs=val_psnrs,
        val_ssims=val_ssims,
        val_sams=val_sams,
        best_psnr=best_psnr,
        train_files=train_files,
        val_files=val_files,
        save_dir=save_dir
    )

    print("\n" + "="*80)
    print("FINAL ARCHITECTURE SUMMARY")
    print("="*80)
    print(f"Model: 4-stage SST Transformer U-Net")
    print(f"Hybrid attention: SST + Spectral + FusedBottleneck")
    print(f"Deep supervision: 3 auxiliary losses")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Best validation PSNR: {best_psnr:.4f} dB")
    print(f"Training epochs: {epoch}")
    print(f"Results directory: {save_dir}")
    print("="*80)

    return model, best_psnr

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) ### for mac m1 support of multiplle workers
    main()