# gssa_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from Conv3d_patch import load_and_process_patches_conv3d  # Use Conv3D processing function

class GSSA(nn.Module):
    """ Guided Spectral Self Attention (GSSA) in "Hybrid Spectral Denoising Transformer with Learnable Query" """
    
    def __init__(self, channel, num_bands, flex=False):
        super().__init__()
        self.channel = channel
        self.num_bands = num_bands
        self.flex = flex
        
        # Learnable query projection
        self.attn_proj = nn.Linear(channel, num_bands)
        # Value projection
        self.value_proj = nn.Linear(channel, channel, bias=False)
        # Final projection
        self.fc = nn.Linear(channel, channel, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            q: Output tensor of shape (B, C, D, H, W)
            attn: Attention weights of shape (B, D, D)
        """
        B, C, D, H, W = x.shape
        residual = x
        
        # Global average pooling across spatial dimensions
        tmp = x.reshape(B, C, D, H * W).mean(-1).permute(0, 2, 1)  # (B, D, C)
        
        # Attention computation
        if self.training:
            if random.random() > 0.5:
                attn = tmp @ tmp.transpose(1, 2)  # Self-attention (B, D, D)
            else:
                attn = self.attn_proj(tmp)  # Learnable query (B, D, num_bands)
        else:
            attn = tmp @ tmp.transpose(1, 2) if self.flex else self.attn_proj(tmp)
        
        # Normalize attention and expand for spatial dimensions
        attn = F.softmax(attn, dim=-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D, D)
        
        # Value transformation and attention application
        v = self.value_proj(rearrange(x, 'b c d h w -> b h w d c'))  # (B, H, W, D, C)
        q = torch.matmul(attn, v)  # Apply attention
        q = self.fc(q)  # Final projection
        q = rearrange(q, 'b h w d c -> b c d h w')  # Back to (B, C, D, H, W)
        q += residual  # Residual connection
        
        return q, attn.squeeze(1).squeeze(1)  # Remove spatial dims from attention


def process_patches_with_gssa(mat_file_path, patch_key='patches', use_gpu=True, batch_size=32):
    """
    Process patches through Conv3D -> GSSA pipeline
    
    Returns:
        original_patches: Original HSI patches
        conv3d_output: Conv3D features  
        gssa_output: GSSA processed features
        attention_maps: GSSA attention weights
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Processing on device: {device}")
    
    # Get Conv3D output
    patch_tensor, conv3d_output, conv3d_model = load_and_process_patches_conv3d(
        mat_file_path, use_gpu=use_gpu, batch_size=batch_size, patch_key=patch_key
    )
    
    if patch_tensor is None or conv3d_output is None:
        raise RuntimeError("Failed to process patches with Conv3D.")
    
    print(f"Conv3D output shape: {conv3d_output.shape}")
    
    # CRITICAL FIX: Rearrange Conv3D output for GSSA
    # Your Conv3D output is (B, C, D, H, W) where D=191 (spectral), H=W=32 (spatial)
    # We need (B, C, D, H, W) format for GSSA, which matches your Conv3D output
    conv3d_for_gssa = conv3d_output  # Already in correct format (B, C, D, H, W)
    print(f"Conv3D for GSSA shape: {conv3d_for_gssa.shape}")
    
    # Use Conv3D's dimensions correctly
    # conv3d_for_gssa shape is (B, C, D, H, W) where D is spectral dimension (191)
    num_channels = conv3d_for_gssa.shape[1]  # C dimension (64)
    num_spectral_bands = conv3d_for_gssa.shape[2]  # D dimension (191)
    
    print(f"GSSA parameters: channels={num_channels}, spectral_bands={num_spectral_bands}")
    
    # Initialize GSSA with correct dimensions
    gssa = GSSA(channel=num_channels, num_bands=num_spectral_bands).to(device)
    gssa.eval()
    
    # Process through GSSA in batches
    gssa_outputs = []
    attention_maps = []
    
    with torch.no_grad():
        for i in range(0, conv3d_for_gssa.shape[0], batch_size):
            batch = conv3d_for_gssa[i:i+batch_size].to(device)
            batch_output, batch_attn = gssa(batch)
            
            gssa_outputs.append(batch_output.cpu())
            attention_maps.append(batch_attn.cpu())
    
    # Combine all batches
    gssa_output = torch.cat(gssa_outputs, dim=0)
    attention_maps = torch.cat(attention_maps, dim=0)
    
    print(f"GSSA output shape: {gssa_output.shape}")
    print(f"Attention maps shape: {attention_maps.shape}")
    
    return patch_tensor.cpu(), conv3d_output.cpu(), gssa_output.cpu(), attention_maps.cpu()


def visualize_input_output(original_patches, conv3d_output, gssa_output, index=0, 
                          save_path="/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/gssa_output/sample_gssa.png"):
    """
    Visualize original patches, Conv3D features, and GSSA output
    """
    
    # Prepare data - all should be in format (B, C, D, H, W)
    original_patch = original_patches[index, 0].numpy()  # (D, H, W) = (191, 32, 32)
    conv3d_patch = conv3d_output[index].numpy()         # (C, D, H, W) = (64, 191, 32, 32)
    gssa_patch = gssa_output[index].numpy()             # (C, D, H, W) = (64, 191, 32, 32)
    
    print(f"Debug - Original patch shape: {original_patch.shape}")
    print(f"Debug - Conv3D patch shape: {conv3d_patch.shape}")
    print(f"Debug - GSSA patch shape: {gssa_patch.shape}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f"Conv3D → GSSA Pipeline Visualization (Sample {index})", fontsize=16)
    
    # Row 1: Original HSI bands
    for i in range(5):
        band_idx = i * original_patch.shape[0] // 5  # shape[0] is spectral dimension (191)
        axes[0, i].imshow(original_patch[band_idx, :, :], cmap='gray')  # (H, W) slice
        axes[0, i].set_title(f"Original Band {band_idx}")
        axes[0, i].axis('off')
    
    # Row 2: Conv3D feature maps (first 5 channels, middle spectral band)
    mid_spectral = conv3d_patch.shape[1] // 2  # shape[1] is spectral dimension (191)
    for i in range(5):
        feature_map = conv3d_patch[i, mid_spectral, :, :]  # (H, W) slice at middle spectral
        axes[1, i].imshow(feature_map, cmap='viridis')
        axes[1, i].set_title(f"Conv3D Ch{i}")
        axes[1, i].axis('off')
    
    # Row 3: GSSA feature maps (first 5 channels, middle spectral band)
    for i in range(5):
        feature_map = gssa_patch[i, mid_spectral, :, :]  # (H, W) slice at middle spectral
        axes[2, i].imshow(feature_map, cmap='plasma')
        axes[2, i].set_title(f"GSSA Ch{i}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive visualization at {save_path}")
    plt.close()


def visualize_attention_matrix(attention_maps, index=0, 
                             save_path="/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/gssa_output/attention_matrix.png"):
    """
    Visualize GSSA attention matrix
    """
    
    attention = attention_maps[index].numpy()  # (D, D)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Attention matrix heatmap
    im = axes[0].imshow(attention, cmap='hot', aspect='auto')
    axes[0].set_title(f'GSSA Attention Matrix (Sample {index})')
    axes[0].set_xlabel('Target Spectral Band')
    axes[0].set_ylabel('Source Spectral Band')
    plt.colorbar(im, ax=axes[0])
    
    # Attention distribution for first few bands
    for i in range(min(5, attention.shape[0])):
        axes[1].plot(attention[i], label=f'Band {i}', alpha=0.7)
    
    axes[1].set_title('Attention Weights for First 5 Bands')
    axes[1].set_xlabel('Target Band')
    axes[1].set_ylabel('Attention Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention visualization at {save_path}")
    plt.close()


def analyze_spectral_enhancement(conv3d_output, gssa_output, index=0,
                               save_path="/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/gssa_output/spectral_enhancement.png"):
    """
    Analyze how GSSA enhances spectral features compared to Conv3D
    """
    
    # Get center pixel spectral profiles
    # conv3d_output: (B, C, D, H, W) where D=191, H=W=32
    # gssa_output: (B, C, D, H, W) where D=191, H=W=32 (same format)
    conv3d_sample = conv3d_output[index].numpy()  # (C, D, H, W) = (64, 191, 32, 32)
    gssa_sample = gssa_output[index].numpy()      # (C, D, H, W) = (64, 191, 32, 32)
    
    print(f"Debug - Conv3D sample shape: {conv3d_sample.shape}")
    print(f"Debug - GSSA sample shape: {gssa_sample.shape}")
    
    # Get center pixel coordinates
    mid_h, mid_w = conv3d_sample.shape[2]//2, conv3d_sample.shape[3]//2  # H, W are at indices 2, 3
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Spectral Enhancement Analysis (Sample {index})', fontsize=14)
    
    # Compare spectral profiles for first 3 channels
    colors = ['red', 'blue', 'green']
    for ch in range(min(3, conv3d_sample.shape[0])):
        # Extract spectral profile at center pixel: (C, D, H, W) -> profile is D dimension
        conv3d_spectral = conv3d_sample[ch, :, mid_h, mid_w]  # Shape: (D,) = (191,)
        gssa_spectral = gssa_sample[ch, :, mid_h, mid_w]      # Shape: (D,) = (191,)
        
        axes[0, 0].plot(conv3d_spectral, color=colors[ch], alpha=0.7, label=f'Conv3D Ch{ch}')
        axes[0, 1].plot(gssa_spectral, color=colors[ch], alpha=0.7, label=f'GSSA Ch{ch}')
    
    axes[0, 0].set_title('Conv3D Spectral Features')
    axes[0, 0].set_xlabel('Spectral Band')
    axes[0, 0].set_ylabel('Feature Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('GSSA Spectral Features')
    axes[0, 1].set_xlabel('Spectral Band')
    axes[0, 1].set_ylabel('Feature Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Enhancement analysis (difference)
    enhancement = gssa_sample[0, :, mid_h, mid_w] - conv3d_sample[0, :, mid_h, mid_w]
    axes[1, 0].plot(enhancement, 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('GSSA Enhancement (GSSA - Conv3D)')
    axes[1, 0].set_xlabel('Spectral Band')
    axes[1, 0].set_ylabel('Feature Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature magnitude comparison - compute L2 norm for each channel
    # For shape (C, D, H, W) = (64, 191, 32, 32), compute norm over (D, H, W) for each channel
    
    # Alternative method: reshape and compute norm
    conv3d_reshaped = conv3d_sample.reshape(conv3d_sample.shape[0], -1)  # (C, D*H*W)
    gssa_reshaped = gssa_sample.reshape(gssa_sample.shape[0], -1)        # (C, D*H*W)
    
    conv3d_magnitude = np.linalg.norm(conv3d_reshaped, axis=1)  # Norm over flattened dimensions
    gssa_magnitude = np.linalg.norm(gssa_reshaped, axis=1)      # Norm over flattened dimensions
    
    print(f"Debug - Conv3D magnitude shape: {conv3d_magnitude.shape}")
    print(f"Debug - GSSA magnitude shape: {gssa_magnitude.shape}")
    
    # Show first 10 channels
    num_channels_to_show = min(10, len(conv3d_magnitude))
    x_pos = np.arange(num_channels_to_show)
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, conv3d_magnitude[:num_channels_to_show], width, label='Conv3D', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, gssa_magnitude[:num_channels_to_show], width, label='GSSA', alpha=0.7)
    axes[1, 1].set_title('Feature Magnitude Comparison')
    axes[1, 1].set_xlabel('Feature Channel')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved spectral enhancement analysis at {save_path}")
    plt.close()


if __name__ == "__main__":
    mat_file_path = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/saved_patches/train_Wash2_patches.mat"
    
    print("🚀 Processing patches through Conv3D → GSSA pipeline...")
    
    # Process patches
    original_patches, conv3d_output, gssa_output, attention_maps = process_patches_with_gssa(
        mat_file_path, patch_key='patches', batch_size=32
    )
    
    print(f"\n📊 Results:")
    print(f"  Original patches: {original_patches.shape}")
    print(f"  Conv3D output: {conv3d_output.shape}")  
    print(f"  GSSA output: {gssa_output.shape}")
    print(f"  Attention maps: {attention_maps.shape}")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    base_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/gssa_output"
    
    # 1. Main pipeline visualization
    visualize_input_output(
        original_patches, conv3d_output, gssa_output, 
        index=0, save_path=f"{base_dir}/pipeline_overview.png"
    )
    
    # 2. Attention matrix visualization
    visualize_attention_matrix(
        attention_maps, index=0, 
        save_path=f"{base_dir}/attention_matrix.png"
    )
    
    # 3. Spectral enhancement analysis
    analyze_spectral_enhancement(
        conv3d_output, gssa_output, index=0,
        save_path=f"{base_dir}/spectral_enhancement.png"
    )
    
    print(f"\n✅ GSSA processing completed successfully!")
    print(f"  📁 All visualizations saved in: {base_dir}")