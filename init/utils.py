############################################ ADDING NOISE TO THE ORIGINAL HSI DATASET ##################################################

import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load HSI from .mat file
# -------------------------------
def load_hsi_from_mat(file_path):
    """
    Loads 3D HSI data from a .mat file.
    Returns:
        - np.ndarray of shape (H, W, C)
        - string key used in .mat file
    """
    data = sio.loadmat(file_path)
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 3:
            return data[key], key
    raise ValueError(f"No valid 3D HSI data found in {file_path}")

# -------------------------------
# 2. Class to inject noise into HSI
# -------------------------------
class HSINoiseInjector:
    def __init__(self, noise_std=0.01):
        """
        Initializes the noise injector.
        Args:
            noise_std: Standard deviation of white Gaussian noise.
        """
        self.noise_std = noise_std

    def add_white_gaussian_noise(self, hsi_tensor):
        """
        Adds white Gaussian noise to an HSI tensor.
        Args:
            hsi_tensor: torch.Tensor of shape (C, H, W) or (1, C, H, W)
        Returns:
            torch.Tensor with added noise
        """
        if hsi_tensor.ndim == 4:
            B, C, H, W = hsi_tensor.shape
        elif hsi_tensor.ndim == 3:
            C, H, W = hsi_tensor.shape
            hsi_tensor = hsi_tensor.unsqueeze(0)
        else:
            raise ValueError("HSI tensor must be of shape (C, H, W) or (1, C, H, W)")

        noise = torch.randn_like(hsi_tensor) * self.noise_std
        noisy_hsi = hsi_tensor + noise
        return noisy_hsi.clamp(0.0, 1.0).squeeze(0)

def load_patches_from_mat(patch_file_path):
    """
    Load patches from a saved .mat file
    Returns:
        patches: torch tensor of shape (num_patches, channels, height, width)
        metadata: dictionary with patch information
    """
    data = sio.loadmat(patch_file_path)
    
    patches = torch.tensor(data['patches']).float()
    coordinates = data['coordinates']
    source_file = str(data['source_file'][0]) if isinstance(data['source_file'], np.ndarray) else data['source_file']
    
    return patches, coordinates, source_file

# -------------------------------
# 3. Visualize HSI
# -------------------------------
def visualize_hsi_comparison(clean_hsi, noisy_hsi, bands=[57, 27, 17], title="HSI Comparison", noise_std=50):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def to_numpy(hsi):
        if isinstance(hsi, torch.Tensor):
            hsi = hsi.detach().cpu().numpy()
        if hsi.ndim == 3 and hsi.shape[0] < hsi.shape[-1]:
            hsi = np.transpose(hsi, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        return hsi

    def normalize_rgb(hsi, bands):
        """
        Normalize selected bands for RGB visualization.
        Args:
            hsi: np.ndarray of shape (H, W, C)
            bands: list of 3 band indices
        Returns:
            np.ndarray of shape (H, W, 3), normalized RGB image
        """
        rgb = hsi[:, :, bands]
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_range = rgb.max(axis=(0, 1), keepdims=True) - rgb_min + 1e-8
        rgb_normalized = (rgb - rgb_min) / rgb_range
        return rgb_normalized

    # Convert tensors to NumPy arrays
    clean_hsi = to_numpy(clean_hsi)
    noisy_hsi = to_numpy(noisy_hsi)

    # Crop to 200×200 if needed
    # clean_hsi = clean_hsi[:200, :200, :]
    # noisy_hsi = noisy_hsi[:200, :200, :]

    # Visualize
    clean_rgb = normalize_rgb(clean_hsi, bands)
    noisy_rgb = normalize_rgb(noisy_hsi, bands)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(clean_rgb)
    axs[0].set_title("Clean HSI RGB")
    axs[0].axis("off")
    axs[0].text(0.5, -0.1, "Noise: 0.00", ha='center', va='center', transform=axs[0].transAxes, fontsize=10)

    axs[1].imshow(noisy_rgb)
    axs[1].set_title("Noisy HSI RGB")
    axs[1].axis("off")
    noise_text = f"Noise: {noise_std:.2f}" if noise_std is not None else "Noise: Unknown"
    axs[1].text(0.5, -0.1, noise_text, ha='center', va='center', transform=axs[1].transAxes, fontsize=10)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

################################################### VISUALIZING THE PATCHES WITH NOISE ################################################### 

def visualize_patch_comparison(clean_patches, noisy_patches, coordinates, max_patches=6, bands=[57, 27, 17]):
    """
    Visualize clean vs noisy patches side by side
    Args:
        clean_patches: torch tensor (num_patches, channels, height, width)
        noisy_patches: torch tensor (num_patches, channels, height, width)
        coordinates: array of (x, y) coordinates
        max_patches: maximum number of patches to display
        bands: RGB band indices for visualization
    """
    num_patches = min(clean_patches.shape[0], max_patches)
    
    fig, axes = plt.subplots(2, num_patches, figsize=(3*num_patches, 6))
    if num_patches == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_patches):
        # Convert patches to numpy and transpose to (H, W, C)
        clean_patch = clean_patches[i].permute(1, 2, 0).numpy()
        noisy_patch = noisy_patches[i].permute(1, 2, 0).numpy()
        
        # Create RGB representations
        clean_rgb = clean_patch[:, :, bands]
        noisy_rgb = noisy_patch[:, :, bands]
        
        # Normalize RGB
        clean_rgb = (clean_rgb - clean_rgb.min()) / (clean_rgb.max() - clean_rgb.min() + 1e-8)
        noisy_rgb = (noisy_rgb - noisy_rgb.min()) / (noisy_rgb.max() - noisy_rgb.min() + 1e-8)
        
        # Plot clean patch
        axes[0, i].imshow(clean_rgb)
        axes[0, i].set_title(f"Clean Patch {i+1}\nPos: {coordinates[i]}", fontsize=10)
        axes[0, i].axis('off')
        
        # Plot noisy patch
        axes[1, i].imshow(noisy_rgb)
        axes[1, i].set_title(f"Noisy Patch {i+1}\nPos: {coordinates[i]}", fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle("Clean vs Noisy HSI Patches Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

#################################################################################################################################################

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Option 1: Process original HSI (your existing code)
    # file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/testset/wdc_crop_test.mat"
    # hsi_data, key = load_hsi_from_mat(file_path)
    # hsi_tensor = torch.tensor(hsi_data).permute(2, 0, 1).float()
    # noise_injector = HSINoiseInjector(noise_std=0.05)
    # noisy_hsi = noise_injector.add_white_gaussian_noise(hsi_tensor)
    # visualize_hsi_comparison(hsi_tensor, noisy_hsi, title="HSI Clean vs Noisy")
    
    # Option 2: Process saved patches (NEW)
    patch_file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/saved_patches/train_Wash2_patches.mat"
    
    # Load patches
    patches, coordinates, source_file = load_patches_from_mat(patch_file_path)
    print(f"Loaded {patches.shape[0]} patches from {source_file}")
    print(f"Patch shape: {patches.shape[1:]} (channels, height, width)")
    
    # Add noise to patches
    noise_injector = HSINoiseInjector(noise_std=0.05)
    noisy_patches = torch.stack([
        noise_injector.add_white_gaussian_noise(patch.unsqueeze(0)).squeeze(0) 
        for patch in patches
    ])
    
    # Visualize clean vs noisy patches
    visualize_patch_comparison(patches, noisy_patches, coordinates, max_patches=6)



# if __name__ == "__main__":
#     # file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset/train_Wash3.mat"
#     file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/testset/wdc_crop_test.mat"
#     # file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset2_wdc/train_Wash_1.mat"

#     # Load
#     hsi_data, key = load_hsi_from_mat(file_path)  # shape: (H, W, C)
#     print(f"Loaded HSI key: {key}, shape: {hsi_data.shape}")

#     # Convert to torch tensor (C, H, W)
#     hsi_tensor = torch.tensor(hsi_data).permute(2, 0, 1).float()

#     # Add noise
#     noise_injector = HSINoiseInjector(noise_std=0.5)
#     noisy_hsi = noise_injector.add_white_gaussian_noise(hsi_tensor)

#     # Visualize
#     visualize_hsi_comparison(hsi_tensor, noisy_hsi, title="HSI Clean vs Noisy")
