# hsi_denoising_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from collections import defaultdict
import json

# Import your existing modules
from Conv3d_patch import extract_and_process_patches_conv3d
from gssa_attention_module import GSSA, process_patches_with_gssa

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


class HSIDenoisingPipeline(nn.Module):
    """
    Complete HSI Denoising Pipeline: Conv3D → GSSA → Feed Forward Block
    """
    def __init__(self, conv3d_channels=64, spectral_bands=191):
        super(HSIDenoisingPipeline, self).__init__()
        
        # GSSA for meaningful spectral band selection
        self.gssa = GSSA(channel=conv3d_channels, num_bands=spectral_bands)
        
        # Feed Forward Block: Progressive refinement and denoising
        self.feed_forward = ResidualConv3DProgressiveRefinement(
            in_channels=conv3d_channels, out_channels=1
        )
    
    def forward(self, conv3d_features):
        """
        Args:
            conv3d_features: (B, C, D, H, W) features from Conv3D extraction
        Returns:
            denoised: (B, 1, D, H, W) denoised patches
            attention_weights: (B, D, D) GSSA attention maps
        """
        # GSSA: Select meaningful spectral bands and model long-range correlations
        gssa_features, attention_weights = self.gssa(conv3d_features)  # (B, C, D, H, W)
        
        # Feed Forward Block: Progressive denoising using aggregated features
        denoised = self.feed_forward(conv3d_features, gssa_features)  # (B, 1, D, H, W)
        
        return denoised, attention_weights


class CombinedLoss(nn.Module):
    """Combined loss for HSI denoising: L1 + L2 + Spectral Consistency"""
    
    def __init__(self, l1_weight=1.0, l2_weight=1.0, spectral_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.spectral_weight = spectral_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
    
    def spectral_consistency_loss(self, pred, target):
        """Compute spectral consistency loss across bands"""
        # Compute spectral gradient along the spectral dimension (D)
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        return F.mse_loss(pred_grad, target_grad)
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        spectral = self.spectral_consistency_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.l2_weight * l2 + 
                     self.spectral_weight * spectral)
        
        return total_loss, {'l1': l1.item(), 'l2': l2.item(), 'spectral': spectral.item()}


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_sam(pred, target):
    """Calculate Spectral Angle Mapper (SAM) - important for HSI evaluation"""
    # Reshape to (B*H*W, D) for spectral angle calculation
    pred_flat = pred.reshape(-1, pred.shape[2])  # (B*H*W, D)
    target_flat = target.reshape(-1, target.shape[2])  # (B*H*W, D)
    
    # Compute cosine similarity
    dot_product = torch.sum(pred_flat * target_flat, dim=1)
    pred_norm = torch.norm(pred_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    
    cos_angle = dot_product / (pred_norm * target_norm + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1, 1)  # Numerical stability
    
    # Spectral angle in radians
    sam_angles = torch.acos(cos_angle)
    return torch.mean(sam_angles)


def visualize_denoising_results(original_patches, conv3d_features, gssa_features, 
                               denoised_patches, attention_maps, epoch, save_dir):
    """Visualize the complete denoising pipeline results"""
    
    # Take first sample from batch
    original = original_patches[0, 0].cpu().numpy()  # (D, H, W)
    conv3d = conv3d_features[0].cpu().numpy()  # (C, D, H, W)
    gssa = gssa_features[0].cpu().numpy()  # (C, D, H, W)
    denoised = denoised_patches[0, 0].cpu().numpy()  # (D, H, W)
    attention = attention_maps[0].cpu().numpy()  # (D, D)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle(f'HSI Denoising Pipeline Results - Epoch {epoch}', fontsize=16)
    
    # Select 5 spectral bands for visualization
    num_bands = original.shape[0]
    band_indices = [i * num_bands // 5 for i in range(5)]
    
    # Row 1: Original noisy patches
    for i, band_idx in enumerate(band_indices):
        axes[0, i].imshow(original[band_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Noisy Band {band_idx}')
        axes[0, i].axis('off')
    
    # Show attention matrix in the last column of row 1
    im = axes[0, 5].imshow(attention, cmap='hot', aspect='auto')
    axes[0, 5].set_title('GSSA Attention Matrix')
    axes[0, 5].set_xlabel('Target Band')
    axes[0, 5].set_ylabel('Source Band')
    plt.colorbar(im, ax=axes[0, 5], shrink=0.6)
    
    # Row 2: Conv3D features (first 5 channels, middle spectral band)
    mid_spectral = conv3d.shape[1] // 2
    for i in range(5):
        feature_map = conv3d[i, mid_spectral, :, :]
        axes[1, i].imshow(feature_map, cmap='viridis')
        axes[1, i].set_title(f'Conv3D Ch{i}')
        axes[1, i].axis('off')
    
    # Show GSSA feature in the last column of row 2
    gssa_feature = gssa[0, mid_spectral, :, :]
    axes[1, 5].imshow(gssa_feature, cmap='plasma')
    axes[1, 5].set_title('GSSA Feature')
    axes[1, 5].axis('off')
    
    # Row 3: Denoised results
    for i, band_idx in enumerate(band_indices):
        axes[2, i].imshow(denoised[band_idx], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Denoised Band {band_idx}')
        axes[2, i].axis('off')
    
    # Show spectral profile comparison in the last column of row 3
    center_h, center_w = original.shape[1]//2, original.shape[2]//2
    original_spectral = original[:, center_h, center_w]
    denoised_spectral = denoised[:, center_h, center_w]
    
    axes[2, 5].plot(original_spectral, 'r-', alpha=0.7, label='Noisy', linewidth=1)
    axes[2, 5].plot(denoised_spectral, 'b-', alpha=0.8, label='Denoised', linewidth=2)
    axes[2, 5].set_title('Center Pixel Spectral Profile')
    axes[2, 5].set_xlabel('Spectral Band')
    axes[2, 5].set_ylabel('Intensity')
    axes[2, 5].legend()
    axes[2, 5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'denoising_results_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Denoising visualization saved: {save_path}")


def plot_training_curves(train_losses, val_losses, train_psnrs, val_psnrs, train_sams, val_sams, save_dir):
    """Plot comprehensive training curves"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR curves
    axes[0, 1].plot(train_psnrs, label='Train PSNR', color='blue', linewidth=2)
    axes[0, 1].plot(val_psnrs, label='Val PSNR', color='red', linewidth=2)
    axes[0, 1].set_title('Training and Validation PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SAM curves
    axes[0, 2].plot(train_sams, label='Train SAM', color='blue', linewidth=2)
    axes[0, 2].plot(val_sams, label='Val SAM', color='red', linewidth=2)
    axes[0, 2].set_title('Training and Validation SAM')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SAM (radians)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss improvement rate
    if len(train_losses) > 10:
        loss_improvement = np.diff(train_losses[-50:]) if len(train_losses) > 50 else np.diff(train_losses)
        axes[1, 0].plot(loss_improvement, color='green', linewidth=2)
        axes[1, 0].set_title('Loss Improvement Rate (Recent)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Change')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # PSNR improvement rate
    if len(train_psnrs) > 10:
        psnr_improvement = np.diff(train_psnrs[-50:]) if len(train_psnrs) > 50 else np.diff(train_psnrs)
        axes[1, 1].plot(psnr_improvement, color='purple', linewidth=2)
        axes[1, 1].set_title('PSNR Improvement Rate (Recent)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PSNR Change (dB)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Training statistics
    axes[1, 2].text(0.1, 0.9, f'Final Train Loss: {train_losses[-1]:.6f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.8, f'Final Val Loss: {val_losses[-1]:.6f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.7, f'Final Train PSNR: {train_psnrs[-1]:.2f} dB', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Final Val PSNR: {val_psnrs[-1]:.2f} dB', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'Final Train SAM: {train_sams[-1]:.4f} rad', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Final Val SAM: {val_sams[-1]:.4f} rad', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].set_title('Final Training Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save curves
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {save_path}")


def train_hsi_denoising_pipeline():
    """Main training function for HSI Denoising Pipeline"""
    
    # Configuration
    config = {
        'train_dir': "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/trainset",
        'save_dir': "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/pipeline_training_results",
        'patch_size': 32,
        'batch_size': 8,  # Reduced for memory efficiency with 3D operations
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 20,
        'visualize_every': 25,
        'k_top': 10,
        'k_mid': 10, 
        'k_rand': 10
    }
    
    print(f"🚀 Starting HSI Denoising Pipeline Training")
    print(f"Pipeline: Conv3D → GSSA → Feed Forward Block")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'visualizations'), exist_ok=True)
    
    # Load patches and extract Conv3D features (this handles noisy patch loading)
    print("📊 Loading noisy patches and extracting Conv3D features...")
    original_patches, conv3d_features, conv3d_model = extract_and_process_patches_conv3d(
        train_dir=config['train_dir'],
        patch_size=config['patch_size'],
        k_top=config['k_top'],
        k_mid=config['k_mid'],
        k_rand=config['k_rand'],
        use_gpu=config['device'] == 'cuda',
        batch_size=config['batch_size']
    )
    
    if original_patches is None or conv3d_features is None:
        raise RuntimeError("Failed to load patches and extract Conv3D features")
    
    print(f"✅ Loaded patches: {original_patches.shape}")
    print(f"✅ Conv3D features: {conv3d_features.shape}")
    
    # Convert to proper format and move to device
    device = torch.device(config['device'])
    original_patches = original_patches.to(device)  # (B, 1, D, H, W) - noisy input
    conv3d_features = conv3d_features.to(device)  # (B, C, D, H, W) - Conv3D features
    
    # Create target (clean) patches - for training, we assume we have ground truth
    # In real scenario, this would come from your clean dataset
    # For now, we'll simulate by using original patches as both noisy and clean
    clean_patches = original_patches.clone()  # This should be replaced with actual clean patches
    
    print(f"💡 Note: Using original patches as both noisy input and clean target")
    print(f"   In real training, load separate clean patches for ground truth")
    
    # Split data for training and validation
    num_samples = original_patches.shape[0]
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    # Create indices for splitting
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"📊 Data split: {train_size} training, {val_size} validation samples")
    
    # Initialize the denoising pipeline
    print("🔧 Initializing HSI Denoising Pipeline...")
    # Get dimensions from Conv3D features
    conv3d_channels = conv3d_features.shape[1]  # C dimension
    spectral_bands = conv3d_features.shape[2]   # D dimension
    
    print(f"   Conv3D channels: {conv3d_channels}, Spectral bands: {spectral_bands}")
    
    model = HSIDenoisingPipeline(
        conv3d_channels=conv3d_channels,
        spectral_bands=spectral_bands
    ).to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(l1_weight=1.0, l2_weight=1.0, spectral_weight=0.5).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training tracking
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_sams, val_sams = [], []
    best_val_psnr = 0
    
    print("🏋️ Starting training loop...")
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss_total = 0
        train_psnr_total = 0
        train_sam_total = 0
        train_batches = 0
        
        # Process training data in batches
        for i in range(0, len(train_indices), config['batch_size']):
            batch_indices = train_indices[i:i + config['batch_size']]
            
            # Get batch data
            batch_conv3d = conv3d_features[batch_indices]  # (batch, C, D, H, W)
            batch_clean = clean_patches[batch_indices]     # (batch, 1, D, H, W)
            batch_original = original_patches[batch_indices]  # (batch, 1, D, H, W)
            
            optimizer.zero_grad()
            
            # Forward pass through pipeline: GSSA → Feed Forward Block
            denoised, attention_weights = model(batch_conv3d)
            
            # Loss calculation
            loss, loss_components = criterion(denoised, batch_clean)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                psnr = calculate_psnr(denoised, batch_clean)
                sam = calculate_sam(denoised, batch_clean)
            
            train_loss_total += loss.item()
            train_psnr_total += psnr.item()
            train_sam_total += sam.item()
            train_batches += 1
            
            # Print progress every 10 batches
            if train_batches % 10 == 0:
                print(f"   Epoch {epoch+1}/{config['num_epochs']}, Batch {train_batches}: "
                      f"Loss={loss.item():.6f}, PSNR={psnr.item():.2f}dB, SAM={sam.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss_total = 0
        val_psnr_total = 0
        val_sam_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_indices), config['batch_size']):
                batch_indices = val_indices[i:i + config['batch_size']]
                
                batch_conv3d = conv3d_features[batch_indices]
                batch_clean = clean_patches[batch_indices]
                batch_original = original_patches[batch_indices]
                
                denoised, attention_weights = model(batch_conv3d)
                loss, _ = criterion(denoised, batch_clean)
                
                psnr = calculate_psnr(denoised, batch_clean)
                sam = calculate_sam(denoised, batch_clean)
                
                val_loss_total += loss.item()
                val_psnr_total += psnr.item()
                val_sam_total += sam.item()
                val_batches += 1
                
                # Save visualization for first validation batch
                if val_batches == 1 and (epoch + 1) % config['visualize_every'] == 0:
                    visualize_denoising_results(
                        batch_original, batch_conv3d, 
                        model.gssa(batch_conv3d)[0],  # Get GSSA features
                        denoised, attention_weights, 
                        epoch + 1, os.path.join(config['save_dir'], 'visualizations')
                    )
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch averages
        avg_train_loss = train_loss_total / train_batches if train_batches > 0 else 0
        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0
        avg_train_psnr = train_psnr_total / train_batches if train_batches > 0 else 0
        avg_val_psnr = val_psnr_total / val_batches if val_batches > 0 else 0
        avg_train_sam = train_sam_total / train_batches if train_batches > 0 else 0
        avg_val_sam = val_sam_total / val_batches if val_batches > 0 else 0
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)
        train_sams.append(avg_train_sam)
        val_sams.append(avg_val_sam)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n📈 Epoch {epoch+1}/{config['num_epochs']} Summary:")
        print(f"   Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"   Train PSNR: {avg_train_psnr:.2f}dB | Val PSNR: {avg_val_psnr:.2f}dB")
        print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_psnr': best_val_psnr,
                'config': config
            }, os.path.join(config['save_dir'], 'checkpoints', 'best_model.pth'))
            print(f"   ⭐ New best model saved! PSNR: {best_val_psnr:.2f}dB")
        
        # Save periodic checkpoints
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_psnrs': train_psnrs,
                'val_psnrs': val_psnrs,
                'config': config
            }, os.path.join(config['save_dir'], 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Final model save
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnrs': train_psnrs,
        'val_psnrs': val_psnrs,
        'config': config
    }, os.path.join(config['save_dir'], 'checkpoints', 'final_model.pth'))
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_psnrs, val_psnrs,
        config['save_dir']
    )
    
    print(f"\n🎉 Training completed!")
    print(f"   Best Validation PSNR: {best_val_psnr:.2f}dB")
    print(f"   Results saved in: {config['save_dir']}")


if __name__ == "__main__":
    train_hsi_denoising_pipeline()