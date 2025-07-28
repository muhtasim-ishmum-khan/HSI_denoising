#### testing Transformer to merge patches after denoising::

############################################ HSI TRANSFORMER MODEL WITH PATCH PROCESSING ##################################################

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ========================================
# 1. TRANSFORMER MODEL CLASSES
# ========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class HSITransformer(nn.Module):
    def __init__(self, patch_size=64, num_channels=191, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, dropout=0.1):
        super(HSITransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.d_model = d_model
        
        # Patch embedding: Convert patch pixels to d_model dimension
        self.patch_embed = nn.Linear(patch_size * patch_size, d_model)
        
        # Channel embedding: Each spectral channel gets its own embedding
        self.channel_embed = nn.Embedding(num_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection back to patch space
        self.output_proj = nn.Linear(d_model, patch_size * patch_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input patches of shape (batch_size, num_channels, patch_size, patch_size)
        Returns:
            Reconstructed patches of same shape
        """
        batch_size, num_channels, patch_h, patch_w = x.shape
        
        # Flatten spatial dimensions: (batch_size, num_channels, patch_size^2)
        # Fix: Use .reshape() instead of .view() to handle non-contiguous tensors
        x_flat = x.reshape(batch_size, num_channels, patch_h * patch_w)
        
        # Embed patches: (batch_size, num_channels, d_model)
        patch_embeddings = self.patch_embed(x_flat)
        
        # Add channel embeddings
        channel_ids = torch.arange(num_channels, device=x.device).unsqueeze(0).expand(batch_size, -1)
        channel_embeddings = self.channel_embed(channel_ids)
        
        # Combine embeddings
        embeddings = patch_embeddings + channel_embeddings
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings)
        
        # Project back to patch space
        output = self.output_proj(embeddings)  # (batch_size, num_channels, patch_size^2)
        
        # Reshape back to patch format - use .reshape() here too
        output = output.reshape(batch_size, num_channels, patch_h, patch_w)
        
        return output

# ========================================
# 2. HSI DATA PROCESSING CLASSES
# ========================================

class HSIDataLoader:
    @staticmethod
    def load_hsi_from_mat(file_path):
        """
        Loads 3D HSI data from a .mat file.
        Returns:
            - np.ndarray of shape (H, W, C)
            - string key used in .mat file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        data = sio.loadmat(file_path)
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 3:
                return data[key], key
        raise ValueError(f"No valid 3D HSI data found in {file_path}")

    @staticmethod
    def load_patches_from_mat(patch_file_path):
        """
        Load patches from a saved .mat file
        Returns:
            patches: torch tensor of shape (num_patches, channels, height, width)
            coordinates: array of (x, y) coordinates
            source_file: source file name
        """
        if not os.path.exists(patch_file_path):
            raise FileNotFoundError(f"Patch file not found: {patch_file_path}")
            
        data = sio.loadmat(patch_file_path)
        
        patches = torch.tensor(data['patches']).float()
        coordinates = data['coordinates']
        source_file = str(data['source_file'][0]) if isinstance(data['source_file'], np.ndarray) else data['source_file']
        
        return patches, coordinates, source_file

class HSIPatchProcessor:
    def __init__(self, patch_size=64, overlap=0):
        """
        Initialize the patch processor.
        Args:
            patch_size: Size of each patch (assuming square patches)
            overlap: Overlap between patches (if any)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def determine_original_size(self, coordinates, patch_size):
        """
        Determine the original HSI dimensions from patch coordinates.
        Args:
            coordinates: Array of (x, y) coordinates
            patch_size: Size of patches
        Returns:
            (height, width) of original HSI
        """
        max_x = np.max(coordinates[:, 0]) + patch_size
        max_y = np.max(coordinates[:, 1]) + patch_size
        return max_y, max_x

    def rejoin_patches(self, patches, coordinates):
        """
        Rejoin patches to form the original HSI.
        Args:
            patches: torch tensor of shape (num_patches, channels, height, width)
            coordinates: array of (x, y) coordinates for each patch
        Returns:
            torch tensor of rejoined HSI of shape (channels, height, width)
        """
        num_patches, channels, patch_h, patch_w = patches.shape
        
        # Determine original dimensions
        orig_h, orig_w = self.determine_original_size(coordinates, patch_h)
        
        print(f"Rejoining {num_patches} patches of size {patch_h}x{patch_w}")
        print(f"Estimated original size: {orig_h}x{orig_w}")
        
        # Initialize the rejoined HSI and overlap counter
        rejoined_hsi = torch.zeros(channels, orig_h, orig_w)
        overlap_counter = torch.zeros(orig_h, orig_w)
        
        # Place each patch in the rejoined HSI
        for i in range(num_patches):
            x, y = coordinates[i]
            x, y = int(x), int(y)
            
            patch = patches[i]  # Shape: (channels, patch_h, patch_w)
            
            # Check bounds
            end_y = min(y + patch_h, orig_h)
            end_x = min(x + patch_w, orig_w)
            patch_h_actual = end_y - y
            patch_w_actual = end_x - x
            
            # Add patch to rejoined HSI
            rejoined_hsi[:, y:end_y, x:end_x] += patch[:, :patch_h_actual, :patch_w_actual]
            overlap_counter[y:end_y, x:end_x] += 1
        
        # Handle overlapping regions by averaging
        overlap_counter[overlap_counter == 0] = 1  # Avoid division by zero
        rejoined_hsi = rejoined_hsi / overlap_counter.unsqueeze(0)
        
        return rejoined_hsi

    def process_patches_with_transformer(self, patches, model):
        """
        Process patches through transformer model.
        Args:
            patches: torch tensor of shape (num_patches, channels, height, width)
            model: HSITransformer model
        Returns:
            processed patches of same shape
        """
        model.eval()
        processed_patches = []
        
        with torch.no_grad():
            for i in range(patches.shape[0]):
                # Process single patch
                patch = patches[i:i+1]  # Keep batch dimension
                processed_patch = model(patch)
                processed_patches.append(processed_patch)
        
        return torch.cat(processed_patches, dim=0)

# ========================================
# 3. VISUALIZATION CLASS
# ========================================

class HSIVisualizer:
    @staticmethod
    def normalize_rgb(hsi, bands):
        """Normalize selected bands for RGB visualization."""
        rgb = hsi[:, :, bands]
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_range = rgb.max(axis=(0, 1), keepdims=True) - rgb_min + 1e-8
        rgb_normalized = (rgb - rgb_min) / rgb_range
        return rgb_normalized

    @staticmethod
    def to_numpy(hsi):
        """Convert tensor to numpy and ensure (H, W, C) format."""
        if isinstance(hsi, torch.Tensor):
            hsi = hsi.detach().cpu().numpy()
        if hsi.ndim == 3 and hsi.shape[0] < hsi.shape[-1]:
            hsi = np.transpose(hsi, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        return hsi

    @classmethod
    def visualize_hsi_comparison(cls, original_hsi, rejoined_hsi, bands=[57, 27, 17], 
                                title="HSI Comparison"):
        """
        Compare original HSI with rejoined HSI from patches.
        Args:
            original_hsi: torch tensor or numpy array of original HSI
            rejoined_hsi: torch tensor of rejoined HSI
            bands: RGB band indices for visualization
            title: Plot title
        """
        # Convert to numpy and ensure (H, W, C) format
        original_np = cls.to_numpy(original_hsi)
        rejoined_np = cls.to_numpy(rejoined_hsi)
        
        print(f"Original HSI shape: {original_np.shape}")
        print(f"Rejoined HSI shape: {rejoined_np.shape}")
        
        # Create RGB representations
        original_rgb = cls.normalize_rgb(original_np, bands)
        rejoined_rgb = cls.normalize_rgb(rejoined_np, bands)
        
        # Calculate difference
        min_h = min(original_rgb.shape[0], rejoined_rgb.shape[0])
        min_w = min(original_rgb.shape[1], rejoined_rgb.shape[1])
        
        original_crop = original_rgb[:min_h, :min_w, :]
        rejoined_crop = rejoined_rgb[:min_h, :min_w, :]
        
        difference = np.abs(original_crop - rejoined_crop)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_crop)
        axes[0].set_title(f"Original HSI\nShape: {original_np.shape}")
        axes[0].axis('off')
        
        axes[1].imshow(rejoined_crop)
        axes[1].set_title(f"Rejoined HSI\nShape: {rejoined_np.shape}")
        axes[1].axis('off')
        
        axes[2].imshow(difference)
        axes[2].set_title(f"Absolute Difference\nMSE: {np.mean(difference**2):.6f}")
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Calculate and print statistics
        mse = np.mean((original_crop - rejoined_crop) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"\nComparison Statistics:")
        print(f"MSE: {mse:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"Max absolute difference: {np.max(difference):.6f}")

# ========================================
# 4. MAIN EXECUTION
# ========================================

# ========================================
# 4. MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # File paths
    patch_file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/init/noisy_patches/train_Wash2_patches_noisy.mat"
    # patch_file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/saved_patches/train_Wash2_patches.mat"
    original_hsi_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset/train_Wash2.mat"
    
    # Initialize components
    data_loader = HSIDataLoader()
    patch_processor = HSIPatchProcessor(patch_size=64)
    visualizer = HSIVisualizer()
    
    print("Loading patches...")
    # Load patches
    patches, coordinates, source_file = data_loader.load_patches_from_mat(patch_file_path)
    print(f"Loaded {patches.shape[0]} patches from {source_file}")
    print(f"Patch shape: {patches.shape[1:]} (channels, height, width)")
    print(f"Coordinate range: X({coordinates[:, 0].min()}-{coordinates[:, 0].max()}), Y({coordinates[:, 1].min()}-{coordinates[:, 1].max()})")
    
    # Initialize transformer model
    print("\nInitializing transformer model...")
    num_channels = patches.shape[1]
    patch_size = patches.shape[2]
    model = HSITransformer(
        patch_size=patch_size, 
        num_channels=num_channels,
        d_model=256,  # Reduced for faster processing
        num_heads=8,
        num_layers=4,  # Reduced for faster processing
        d_ff=1024,
        dropout=0.1
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Process patches through transformer (identity operation for now)
    print("\nProcessing patches through transformer...")
    processed_patches = patch_processor.process_patches_with_transformer(patches, model)
    
    print("\nRejoining processed patches...")
    # Rejoin processed patches
    rejoined_hsi = patch_processor.rejoin_patches(processed_patches, coordinates)
    print(f"Rejoined HSI shape before transpose: {rejoined_hsi.shape}")
    
    # TRANSPOSE FIX: Convert from (C, H, W) to (H, W, C) format
    rejoined_hsi = rejoined_hsi.permute(2, 1, 0)
    print(f"Rejoined HSI shape after transpose: {rejoined_hsi.shape}")
    
    print("\nLoading original HSI...")
    # Load original HSI
    original_hsi_data, key = data_loader.load_hsi_from_mat(original_hsi_path)
    print(f"Loaded original HSI with key '{key}', shape: {original_hsi_data.shape}")
    
    # Convert original HSI to tensor format (keep H, W, C format)
    original_hsi_tensor = torch.tensor(original_hsi_data).float()
    
    print("\nComparing original and rejoined HSI...")
    # Visualize comparison
    visualizer.visualize_hsi_comparison(
        original_hsi_tensor, 
        rejoined_hsi, 
        bands=[57, 27, 17], 
        title="Original vs Transformer-Processed Rejoined HSI"
    )