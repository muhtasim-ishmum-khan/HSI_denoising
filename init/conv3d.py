import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from generate_patches_restormer import generate_patches_from_mat

# The SpectralSpatialConv3D class is a custom neural network module o process 3D HSI data.
# Simply, it looks at a 3D block of data and transforms it into a new set of data that highlights important patterns or features.
# The module uses a dual-branch architecture, meaning it processes the input data in two parallel branches with slightly different processing techniques,
# then combines the results. This is for capturing different types of patterns in the data.
# This dual-branch architecture was followed from the paper: "3D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising"
class SpectralSpatialConv3D(nn.Module):
    # in_channels=1: The number of input channels, treating each hyperspectral patch as a single 3D volume.
    # out_channels=n means the model produces n different feature maps to capture a diverse set of patterns in the HSI patches. 
    # For example, one feature map might highlight edges in the spatial dimensions, while another captures changes across spectral bands.
    def __init__(self, in_channels=1, out_channels=64, dropout=0.2):
        super(SpectralSpatialConv3D, self).__init__()

        # Layers in the first branch with ReLU.

        # 3D convolution layer that takes in_channels as input and produces 32 feature maps (each highlighting different features).
        # kernel_size=3 means it uses a 3x3x3 cube to scan the data (covering height, width, and spectral dimensions).
        self.branch1_conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        # Batch normalization normalizes the output of the convolution to have a mean of 0 and a standard deviation of 1, stabilizes training
        self.branch1_bn1 = nn.BatchNorm3d(32)
        # in 32 features maps from previous layer and out n(out_channels) feature maps, refining the features further.
        self.branch1_conv2 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.branch1_bn2 = nn.BatchNorm3d(out_channels)

        # Layers in the first branch with GELU.
        self.branch2_conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.branch2_bn1 = nn.BatchNorm3d(32)
        self.branch2_conv2 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.branch2_bn2 = nn.BatchNorm3d(out_channels)
        # The dropout rate, which is a technique to prevent overfitting by randomly turning off x% of the neurons during training.
        self.dropout = nn.Dropout3d(p=dropout)
    
    # The forward method defines how the input data flows through the layers to produce the output.
    def forward(self, x):
        # Branch 1 with ReLU activation.
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        # Branch 2 with GELU activation.
        x2 = F.gelu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = F.gelu(self.branch2_bn2(self.branch2_conv2(x2)))

        # Aggregation of the output of two branches (element-wise addition)
        x_fused = x1 + x2
        x_fused = self.dropout(x_fused)
        return x_fused

# The extract_and_process_patches function takes HSI data,
# extracts smaller patches from these images,
# processes them through a Conv3D model to extract features,
# and returns both the input patches and the processed outputs.
def extract_and_process_patches_conv3d(train_dir, patch_size=32, k_top=10, k_mid=10, k_rand=10, use_gpu=True, batch_size=32):
    # looks in the train_dir folder and finds all files ending with .mat
    mat_files = [f for f in os.listdir(train_dir) if f.endswith(".mat")]
    # list to store the patches extracted from these files.
    all_patches = []
    # It calls generate_patches_from_mat to extract patches from the HSI data.
    # The extracted patches are added to the all_patches list.
    for file_name in mat_files:
        file_path = os.path.join(train_dir, file_name)
        try:
            patches, _, _ = generate_patches_from_mat(file_path, k_top=k_top, k_mid=k_mid, k_rand=k_rand, patch_size=patch_size) 
            all_patches.extend(patches)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    # If no patches were extracted, this prevents the function from proceeding with empty data.
    if len(all_patches) == 0:
        print("No patches found.")
        return None, None, None
    # combines all patches into a single PyTorch tensor. Each patch is a 3D array (height, width, spectral bands), 
    # so stacking them creates a 4D tensor of shape (num_patches, height, width, spectral_bands).
    # .unsqueeze(1) adds an extra dimension, making it (num_patches, 1, height, width, spectral_bands). 
    # The extra dimension (channel) is needed because Conv3D expects a channel dimension
    patch_tensor = torch.stack(all_patches).unsqueeze(1)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #
    model = SpectralSpatialConv3D(in_channels=1, out_channels=64).to(device)
    # Sets the model to evaluation mode. Because, During inference, we want the full model capacity, so dropout should be disabled.
    model.eval()

    #
    outputs = []
    # torch.no_grad() disables gradient computation.
    with torch.no_grad():
        # Loops through the patch_tensor in chunks of batches.
        for i in range(0, patch_tensor.shape[0], batch_size):
            # Extracts a subset of patches (i to batch_size).
            batch = patch_tensor[i:i+batch_size].to(device)
            # Passes the batch through the model to get feature maps.
            batch_output = model(batch)
            # Moves the output back to CPU and stores it in outputs.
            outputs.append(batch_output.cpu())
    # Combines all batch outputs into a single tensor using torch.cat along dimension 0 (the batch dimension).
    output_tensor = torch.cat(outputs, dim=0)
    return patch_tensor, output_tensor, model

# The visualize_feature_maps function takes input and output tensors from a neural network, 
# selects one specific patch, and creates a side-by-side visualization.
def visualize_feature_maps(input_tensor, output_tensor, index=0, file="sample", x=0, y=0):
    input_patch = input_tensor[index, 0].cpu().numpy()
    output_patch = output_tensor[index].cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Input Spectral Bands vs Output Feature Maps", fontsize=14)

    for i in range(5):
        band_idx = i * input_patch.shape[0] // 5
        axes[0, i].imshow(input_patch[band_idx], cmap='gray')
        axes[0, i].set_title(f"Input Band {band_idx}")
        axes[0, i].axis('off')

    for i in range(5):
        axes[1, i].imshow(output_patch[i, output_patch.shape[1]//2], cmap='viridis')
        axes[1, i].set_title(f"Feature Map {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    save_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/init/conv_3d"
    os.makedirs(save_dir, exist_ok=True)

    # Build a base filename
    base_filename = f"{file}_patch_{x}_{y}"
    save_path = os.path.join(save_dir, f"{base_filename}.png")

    # Add a counter if the file already exists
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(save_dir, f"{base_filename}_{counter}.png")
        counter += 1

    plt.savefig(save_path)
    print(f"Patch visualization saved at: {save_path}")
    # plt.show()

if __name__ == "__main__":
    train_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/trainset"
    patch_size = 32

    patch_tensor, output, model = extract_and_process_patches_conv3d(train_dir, patch_size=patch_size, batch_size=32)

    if patch_tensor is not None and output is not None:
        print(f"Input shape: {patch_tensor.shape}")
        print(f"Output shape: {output.shape}")
        visualize_feature_maps(patch_tensor, output, index=0, file="sample", x=0, y=0)
    else:
        print("Patch processing failed.")

    