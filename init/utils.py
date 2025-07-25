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



# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset/train_Wash3.mat"
    file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/testset/wdc_crop_test.mat"
    # file_path = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset2_wdc/train_Wash_1.mat"

    # Load
    hsi_data, key = load_hsi_from_mat(file_path)  # shape: (H, W, C)
    print(f"Loaded HSI key: {key}, shape: {hsi_data.shape}")

    # Convert to torch tensor (C, H, W)
    hsi_tensor = torch.tensor(hsi_data).permute(2, 0, 1).float()

    # Add noise
    noise_injector = HSINoiseInjector(noise_std=0.5)
    noisy_hsi = noise_injector.add_white_gaussian_noise(hsi_tensor)

    # Visualize
    visualize_hsi_comparison(hsi_tensor, noisy_hsi, title="HSI Clean vs Noisy")
