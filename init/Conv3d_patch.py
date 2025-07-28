import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import psutil
import time
from tqdm import tqdm

class SpectralSpatialConv3D(nn.Module):
    """
    3D CNN with dual-branch architecture for hyperspectral image feature extraction.
    Uses ReLU and GELU activation functions in parallel branches.
    """
    def __init__(self, in_channels=1, out_channels=64, dropout=0.2):
        super(SpectralSpatialConv3D, self).__init__()

        # Branch 1 with ReLU activation
        self.branch1_conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.branch1_bn1 = nn.BatchNorm3d(32)
        self.branch1_conv2 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.branch1_bn2 = nn.BatchNorm3d(out_channels)

        # Branch 2 with GELU activation
        self.branch2_conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.branch2_bn1 = nn.BatchNorm3d(32)
        self.branch2_conv2 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.branch2_bn2 = nn.BatchNorm3d(out_channels)
        
        self.dropout = nn.Dropout3d(p=dropout)
    
    def forward(self, x):
        # Branch 1 with ReLU
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        
        # Branch 2 with GELU
        x2 = F.gelu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = F.gelu(self.branch2_bn2(self.branch2_conv2(x2)))

        # Fuse branches and apply dropout
        x_fused = x1 + x2
        x_fused = self.dropout(x_fused)
        return x_fused

def load_patches_from_mat(mat_file_path, patch_key='patches'):
    """
    Load patches from a .mat file and convert to PyTorch tensor.
    
    Args:
        mat_file_path (str): Path to .mat file
        patch_key (str): Key containing patch data
    
    Returns:
        torch.Tensor: Patches in format (N, 1, H, W, C)
    """
    try:
        print(f"Loading patches from: {mat_file_path}")
        mat_data = sio.loadmat(mat_file_path)
        
        # Try to find patches with the specified key
        if patch_key in mat_data:
            patches = mat_data[patch_key]
        else:
            # Try common alternative keys
            possible_keys = ['patches', 'data', 'patch_data', 'hsi_patches', 'X']
            found_key = None
            for key in possible_keys:
                if key in mat_data and not key.startswith('__'):
                    found_key = key
                    break
            
            if found_key:
                patches = mat_data[found_key]
                print(f"Found patches under key: '{found_key}'")
            else:
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                raise KeyError(f"No patch data found. Available keys: {available_keys}")
        
        print(f"Original patches shape: {patches.shape}")
        
        # Handle different input formats
        if len(patches.shape) == 4:
            # Assume format: (N, H, W, C) -> convert to (N, 1, H, W, C)
            if patches.shape[-1] < patches.shape[1]:  # Spectral bands likely in last dimension
                patches_tensor = torch.from_numpy(patches).float()
                patches_tensor = patches_tensor.unsqueeze(1)  # Add channel dimension
            else:
                # Format might be (N, C, H, W) -> convert to (N, 1, H, W, C)
                patches = np.transpose(patches, (0, 2, 3, 1))
                patches_tensor = torch.from_numpy(patches).float()
                patches_tensor = patches_tensor.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected patch shape: {patches.shape}")
        
        print(f"Converted tensor shape: {patches_tensor.shape}")
        return patches_tensor
        
    except Exception as e:
        print(f"Error loading patches: {e}")
        return None

def get_memory_usage():
    """Get current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB

def train_conv3d_model(patches_tensor, num_epochs=50, batch_size=16, learning_rate=0.001, use_gpu=True):
    """
    Train the Conv3D model using self-supervised learning approach.
    
    Args:
        patches_tensor: Input patches tensor
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        use_gpu: Whether to use GPU
    
    Returns:
        model: Trained model
        loss_history: Training loss history
        memory_history: Memory usage history
    """
    device = torch.device(
    "cuda" if use_gpu and torch.cuda.is_available() else
    "mps" if use_gpu and torch.backends.mps.is_available() else
    "cpu"
)
    print(f"Training on device: {device}")
    
    # Initialize model
    model = SpectralSpatialConv3D(in_channels=1, out_channels=64).to(device)
    
    # Define loss function (using MSE for reconstruction-like task)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For self-supervised learning, we'll use the input as both input and target
    # The model learns to extract meaningful features
    patches_tensor = patches_tensor.to(device)
    
    # Training history
    loss_history = []
    memory_history = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset size: {patches_tensor.shape[0]} patches")
    print(f"Batch size: {batch_size}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create random indices for batch sampling
        indices = torch.randperm(patches_tensor.shape[0])
        
        # Progress bar for batches
        batch_pbar = tqdm(range(0, patches_tensor.shape[0], batch_size), 
                         desc=f"Epoch {epoch+1}/{num_epochs}", 
                         leave=False)
        
        for i in batch_pbar:
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            batch_data = patches_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            features = model(batch_data)
            
            # For self-supervised learning, we can use different objectives
            # Here we use feature consistency loss
            target = batch_data.squeeze(1)  # Remove channel dimension for target
            
            # Reduce feature dimensions to match target
            features_reduced = torch.mean(features, dim=1)  # Average across feature channels
            
            loss = criterion(features_reduced, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Monitor memory usage
        memory_usage = get_memory_usage()
        memory_history.append(memory_usage)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - RAM: {memory_usage:.2f} GB")
        
        # Clear cache to manage memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Training completed!")
    return model, loss_history, memory_history

def visualize_results(original_patches, model, device, save_dir="/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/conv_3d"):
    """
    Visualize original patches and extracted features side by side.
    
    Args:
        original_patches: Original patch tensor
        model: Trained model
        device: Computing device
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Select a few random patches for visualization
        num_samples = min(5, original_patches.shape[0])
        sample_indices = torch.randperm(original_patches.shape[0])[:num_samples]
        
        for idx, sample_idx in enumerate(sample_indices):
            sample_patch = original_patches[sample_idx:sample_idx+1].to(device)
            features = model(sample_patch)
            
            # Move to CPU for visualization
            original = sample_patch[0, 0].cpu().numpy()  # Shape: (H, W, C)
            feature_maps = features[0].cpu().numpy()  # Shape: (64, H, W, C)
            
            # Create visualization
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            fig.suptitle(f"Original Patch vs Feature Maps - Sample {idx+1}", fontsize=16)
            
            # Show original patch (5 spectral bands)
            for i in range(5):
                band_idx = i * original.shape[2] // 5
                axes[0, i].imshow(original[:, :, band_idx], cmap='gray')
                axes[0, i].set_title(f"Original Band {band_idx}")
                axes[0, i].axis('off')
            
            # Show first 5 feature maps (middle slice)
            for i in range(5):
                mid_slice = feature_maps.shape[2] // 2
                axes[1, i].imshow(feature_maps[i, :, :, mid_slice], cmap='viridis')
                axes[1, i].set_title(f"Feature Map {i}")
                axes[1, i].axis('off')
            
            # Show next 5 feature maps
            for i in range(5):
                mid_slice = feature_maps.shape[2] // 2
                axes[2, i].imshow(feature_maps[i+5, :, :, mid_slice], cmap='plasma')
                axes[2, i].set_title(f"Feature Map {i+5}")
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            save_path = os.path.join(save_dir, f"patch_features_sample_{idx+1}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
            plt.close()

def plot_training_history(loss_history, memory_history, save_dir="/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/conv_3d"):
    """Plot training loss and memory usage over epochs"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot memory usage
    ax2.plot(memory_history, 'r-', linewidth=2)
    ax2.set_title('RAM Usage Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RAM Usage (GB)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved: {save_path}")
    plt.show()

def main():
    """Main function to orchestrate the training process"""
    
    # Configuration
    mat_file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/saved_patches/train_Wash2_patches"
    save_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/conv3d_results"
    
    # Training parameters
    num_epochs = 30
    batch_size = 16
    learning_rate = 0.001
    use_gpu = True
    
    print("=" * 60)
    print("HSI Conv3D Feature Extraction Training")
    print("=" * 60)
    
    # Load patches
    print("\n1. Loading patches from .mat file...")
    patches_tensor = load_patches_from_mat(mat_file_path)
    
    if patches_tensor is None:
        print("Failed to load patches. Exiting...")
        return
    
    print(f"Successfully loaded {patches_tensor.shape[0]} patches")
    print(f"Patch dimensions: {patches_tensor.shape[1:]}")
    
    # Train model
    print("\n2. Starting training...")
    initial_memory = get_memory_usage()
    print(f"Initial RAM usage: {initial_memory:.2f} GB")
    
    model, loss_history, memory_history = train_conv3d_model(
        patches_tensor, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_gpu=use_gpu
    )
    
    final_memory = get_memory_usage()
    print(f"Final RAM usage: {final_memory:.2f} GB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
    
    # Visualize results
    print("\n3. Generating visualizations...")
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    visualize_results(patches_tensor, model, device, save_dir)
    
    # Plot training history
    print("\n4. Plotting training history...")
    plot_training_history(loss_history, memory_history, save_dir)
    
    # Save trained model
    model_path = os.path.join(save_dir, "trained_conv3d_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training and visualization completed!")
    print(f"Results saved in: {save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()