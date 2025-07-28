# Importing necessary libraries.
import torch
import torch.nn as nn # for accessing neuralnetwork module in pytorch.
import torch.nn.functional as F # for imporing APIs and functions for building neural networks.
from einops import rearrange # for rearranging or shuffling dimensions.
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


# The MDTA (Multi-Dconv Head Transposed Attention) is the core block for computing attention scores.
class MDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # The num_heads parameter determines how many attention heads the module will use. Each head focuses on different aspects of the image.
        self.num_heads = num_heads
        # In attention, we compute dot products between vectors. If the vectors have large dimensions, the dot products can become very large.
        # Scaling keeps the values in a reasonable range.
        self.scale = (dim // num_heads) ** -0.5

        # Query, Key, Value (qkv) Convolution layer takes an input dim channels and outputs dim * 3 channels.
        # These channels are split into query, key, and value, each with dim channels.
        # bias = false to reduce parameters slightly and 1 means kernel size (1x1 convolution).
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        # In hyperspectral images, adjacent pixels often have correlated spectral signatures.
        # The self.dwconv layer applies a 3x3 filter to each channel of q, k, and v, incorporating information from the 3x3 neighborhood around each pixel.
        # This adds local spatial context and enriches q,k,v
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)

        # The proj convolution takes the attention output (dim channels) and maps it back to dim channels.
        # The self.proj layer acts like a post-processor that refines raw attention output into a more coherent and optimized representation.
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # This applies the softmax function along the last dimension to normalize attention scores into probabilities.
        self.softmax = nn.Softmax(dim=-1)

        self.latest_attn_map = None

    # The forward method defines how the input data flows through the MDTA module
    def forward(self, x):
        # The input x is a tensor with shape (B, C, H, W)
        B, C, H, W = x.shape

        # Generate q, k, v
        qkv = self.qkv(x)
        # Adds local spatial context by considering the 3x3 neighborhood around each pixel, enhancing q, k, and v with information from nearby pixels.
        qkv = self.dwconv(qkv)
        # the qkv convolution transforms the input x from dim channels to dim * 3 channels.
        # The chunk operation splits the tensor into 3 equal parts (querry, key, values).
        # dim=1 is the index of channel dimension.
        q, k, v = qkv.chunk(3, dim=1)
        # Divides the channel dimension C into num_heads to harness the multi head attention mechanism.
        head_dim = C // self.num_heads
        assert head_dim * self.num_heads == C, "C must be divisible by num_heads"

        # Each of q, k, and v (shape (B, C, H, W)) is reshaped to (B, num_heads, head_dim, H*W) and then transposed to (B, num_heads, H*W, head_dim).
        # This reshaping is necessary for multi-head attention, as now model knows how many channel a head will handle.
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        # Applies L2 normalization that Ensures that query and key vectors have unit length,
        # which stabilizes the dot-product attention computation by preventing large values.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Computes the dot product between q (shape (B, h, HW, head_dim)) and the transposed k (shape (B, h, head_dim, HW)).
        # The result is an attention matrix of shape (B, h, HW, HW), where each entry represents the similarity between two pixels.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, HW, HW)
        attn = self.softmax(attn)

        # Save attention map
        # For patch extraction, we need a single map to decide which pixels are most important so ween need to combine output of the heads.
        self.latest_attn_map = attn.mean(dim=0)  # shape: (h, HW, HW)

        # Multiplies the attention matrix ((B, h, HW, HW)) with the value tensor ((B, h, HW, head_dim)).
        # The result is a weighted sum of values, where weights come from the attention scores.
        out = torch.matmul(attn, v)
        # The tensor reshaped back to the original image-like format that ensures the output matches the input shape of the MDTA module.
        out = out.transpose(2, 3).reshape(B, C, H, W)
        # It helps the model learn how to best integrate the information from the multi-head attention by refining the attention output.
        # Its like fine tuning a dish one last time before serving.
        return self.proj(out)



# Core architecture of the Restormer.
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Creates an instance of the MDTA class that is the core attention block.
        self.mdta = MDTA(dim, num_heads)
        # creates a Layer Normalization module.
        # [dim, 1, 1] means that normalization is applied per pixel across the channel dimension.
        # As attention typically computes relationships between individual pixels, per pixel normalization is logical here.
        self.norm = nn.LayerNorm(dim)

    # Defines how data flows through the block during processing.
    def forward(self, x):
        # save the original input.
        residual = x
        # Rearranges the dimensions of the input tensor x from (B, C, H, W) to (B, H, W, C).
        # Because xpects the channel dimension (C) to be the last dimension.
        x = x.permute(0, 2, 3, 1)
        # LayerNorm stabilizes the input to the MDTA module by ensuring the feature map’s values are in a consistent range.
        # As Hyperspectral images have high-dimensional data and values can vary widely, this is necessary.
        x = self.norm(x)
        # Rearranges the tensor back from (B, H, W, C) to (B, C, H, W) for input to MDTA
        x = x.permute(0, 3, 1, 2)
        # Passes the normalized feature map through the MDTA module.
        x = self.mdta(x)
        # This is a residual or skip connection. The residual connection ensures the model retains the original HSI information while adding improvements stably.
        return residual + x

class RestormerHSI(nn.Module):
    # in_channels: The number of spectral bands in the input HSI.
    def __init__(self, in_channels, dim=48, num_heads=6):
        super().__init__()
        # The convolution layer works as a local feature extractor.
        # As, attention block captures global relationship among pixels we need something to extract local information (edges, textures)
        # Simply, this is a feature extractor that converts the raw HSI data into a more manageable set of (dim = n) feature channels.
        # Each channel captures different patterns (like edges, textures) across the spectral bands.
        self.embedding = nn.Conv2d(in_channels, dim, 3, padding=1)
        # encoder is like the brain of the model. It processes the initial features to highlight important parts of the HSI.
        self.encoder = nn.Sequential(
            RestormerBlock(dim, num_heads),
            RestormerBlock(dim, num_heads),
        )
        # This layer is like a reconstructor that translates the internal features (dim=n channels) back into the original HSI format (original n channels).
        self.output = nn.Conv2d(dim, in_channels, 3, padding=1)

    # This forward layer defines how the data flows through the block.
    def forward(self, x):
        # data flow: input -> local feature extraction and channel conversion -> important patterns attented -> reconstruct to original shape
        features = self.embedding(x)
        features = self.encoder(features)
        return self.output(features)

    # This method retrieves the attention map generated by the last RestormerBlock in the encoder.
    def get_attention_map(self):
        # Accesses the last restormer block by [-1] then go to the corresponding mdta block to get the attention map.
        return self.encoder[-1].mdta.latest_attn_map




        
  ########################################## ADDED PARTS TO EXTRACT THE PATCHES AS MAT FILES #############################################
class HSIPatchStorage:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_patches_as_mat(self, patches, metadata, source_file):
        """
        Save patches from a single HSI file as a .mat file
        
        Args:
            patches: List of patch tensors
            metadata: List of metadata tuples (file_name, x, y, band_idx)
            source_file: Original .mat file name
        """
        if not patches:
            print(f"No patches to save for {source_file}")
            return
            
        # Convert patches to numpy array
        patches_array = torch.stack(patches).numpy()  # Shape: (num_patches, channels, height, width)
        
        # Extract coordinates and other info
        coordinates = []
        band_indices = []
        
        for meta in metadata:
            file_name, x, y, band_idx = meta
            coordinates.append([x, y])
            band_indices.append(band_idx)
        
        # Prepare data dictionary for .mat file
        mat_data = {
            'patches': patches_array,
            'coordinates': np.array(coordinates),
            'band_indices': np.array(band_indices),
            'source_file': source_file,
            'num_patches': len(patches),
            'patch_shape': patches_array.shape[1:],  # (channels, height, width)
            'creation_timestamp': str(np.datetime64('now'))
        }
        
        # Create output filename
        base_name = os.path.splitext(source_file)[0]
        output_file = os.path.join(self.output_dir, f"{base_name}_patches.mat")
        
        # Save as .mat file
        sio.savemat(output_file, mat_data)
        
        print(f"Saved {len(patches)} patches from {source_file} to {output_file}")
        return output_file
    
    def load_patches_from_mat(self, patch_file):
        """
        Load patches from a .mat file
        
        Args:
            patch_file: Path to the patches .mat file
            
        Returns:
            patches: torch tensor of patches
            metadata: dictionary with patch information
        """
        data = sio.loadmat(patch_file)
        
        patches = torch.tensor(data['patches']).float()
        
        metadata = {
            'coordinates': data['coordinates'],
            'band_indices': data['band_indices'],
            'source_file': str(data['source_file'][0]) if isinstance(data['source_file'], np.ndarray) else data['source_file'],
            'num_patches': int(data['num_patches']),
            'patch_shape': tuple(data['patch_shape'].flatten()),
            'creation_timestamp': str(data['creation_timestamp'][0]) if 'creation_timestamp' in data else 'Unknown'
        }
        
        return patches, metadata
####################################################################################################################################





# This function is designed to identify and extract the most important regions or patches from a HSI based on an attention map.
def extract_patches(
    x,
    attn_map,
    k_top=10,
    k_mid=10,
    k_rand=10,
    patch_size=32,
    file_name=None,
    band_idx=None
):
    B, C, H, W = x.shape
    patch_list = []
    coords = []
    metadata = []

    total_patches = k_top + k_mid + k_rand

    for b in range(B):
        attn = attn_map[b].clone()  # (H, W)
        flat_attn = attn.view(-1)
        num_pixels = flat_attn.shape[0]

        # Sort attention values
        sorted_vals, sorted_indices = torch.sort(flat_attn, descending=True)

        # -- Top-k attention patches
        top_indices = sorted_indices[:k_top]

        # -- Mid-k attention patches (from middle of the sorted list)
        mid_start = num_pixels // 3
        mid_end = 2 * num_pixels // 3
        mid_pool = sorted_indices[mid_start:mid_end]
        mid_indices = mid_pool[torch.randint(0, len(mid_pool), (k_mid,))]

        # -- Random-k patches (uniformly random)
        rand_indices = torch.randint(0, num_pixels, (k_rand,))

        combined_indices = torch.cat([top_indices, mid_indices, rand_indices])

        for idx in combined_indices:
            y, x_idx = idx // W, idx % W
            y1 = max(0, y - (patch_size // 2))
            x1 = max(0, x_idx - (patch_size // 2))
            y2 = min(H, y1 + patch_size)
            x2 = min(W, x1 + patch_size)

            # Fix sizes if we're near the edges
            y1 = y2 - patch_size if y2 - y1 < patch_size else y1
            x1 = x2 - patch_size if x2 - x1 < patch_size else x1

            # Final patch
            patch = x[b, :, y1:y2, x1:x2]
            patch_list.append(patch)
            coords.append((y1, x1))
            metadata.append((
                file_name,
                y1,
                x1,
                band_idx if band_idx is not None else -1
            ))

    return patch_list, coords, metadata


# Automated .mat loader
def load_hsi_from_mat(file_path):
    data = sio.loadmat(file_path)
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 3:
            return data[key], key
    raise ValueError(f"No valid 3D HSI data found in {file_path}")


#Patch generation pipeline
def generate_patches_from_mat(file_path, k_top, k_mid, k_rand, patch_size=32):
    # Load the hyperspectral image and key from the .mat file
    hsi_data, key_used = load_hsi_from_mat(file_path)

    # Convert HSI data to tensor with shape (B, C, H, W)
    hsi_tensor = torch.tensor(hsi_data).permute(2, 0, 1).unsqueeze(0).float()

    # Initialize RestormerHSI model
    model = RestormerHSI(in_channels=hsi_tensor.shape[1])

    # Disable gradient computation
    with torch.no_grad():
        _ = model(hsi_tensor)
        attn_map = model.get_attention_map()  # shape: (num_heads, H*W, H*W)

    # Mean over heads -> (H*W, H*W)
    mean_attn = attn_map.mean(dim=0)

    # Mean over keys (second dimension) to get per-pixel saliency
    saliency = mean_attn.mean(dim=0)  # shape: (H*W,)

    # Recover H and W from input tensor
    _, _, H, W = hsi_tensor.shape
    expected_len = H * W
    if saliency.shape[0] != expected_len:
        raise ValueError(f"Expected saliency length {expected_len}, but got {saliency.shape[0]}")

    # Reshape attention saliency into 2D map
    attn_2d = saliency.view(H, W)

    # Extract patches based on the attention map
    patches, coords, metadata = extract_patches(
        hsi_tensor,
        attn_2d.unsqueeze(0),  # shape: (1, H, W)
        k_top=k_top,
        k_mid=k_mid,
        k_rand=k_rand,
        patch_size=patch_size,
        file_name=os.path.basename(file_path),
        band_idx=hsi_data.shape[2] // 2
    )

    return patches, coords, metadata


# Visualization
def visualize_patch(train_dir, patch_img, file, x, y, j, patch_size):
    mat_path = os.path.join(train_dir, file)
    full_data, _ = load_hsi_from_mat(mat_path)

    w, h, depth = full_data.shape
    K = patch_img.shape[0] - 1
    nz = depth + K

    padded = np.zeros((w, h, nz), dtype=full_data.dtype)
    padded[:, :, 0:K//2] = full_data[:, :, (K//2):0:-1]
    padded[:, :, K//2:nz - K//2] = full_data
    padded[:, :, nz - K//2:] = full_data[:, :, -2:-(K//2 + 2):-1]

    bands = [j-1, j, j+1]
    bands = [max(K//2, min(b, nz-K//2-1)) for b in bands]
    rgb_image = padded[:, :, bands]
    rgb_norm = (rgb_image - rgb_image.min(axis=(0,1))) / (np.ptp(rgb_image, axis=(0,1)) + 1e-6)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(rgb_norm)
    rect = mpatches.Rectangle((y, x), patch_size, patch_size, linewidth=2, edgecolor='yellow', facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set_title(f"Full Image with Patch Location\nFile: {file}")
    axs[0].axis('off')

    axs[1].imshow(patch[0].numpy(), cmap='gray')
    axs[1].set_title("Spectral Band 1 of Patch")
    axs[1].axis('off')

    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(
        #"/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/init/patch_img_MK",
        "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/init/patch_img"
        f"{file}_patch_{x}_{y}.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Patch visualization saved at: {save_path}")
    plt.close()


# Main Execution Block
# if __name__ == "__main__":
#     train_dir = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset"
#     patch_size = 32

#     total_patches = 0
#     all_patches = []
#     all_metadata = []

#     mat_files = [f for f in os.listdir(train_dir) if f.endswith(".mat")]


#     for file_name in mat_files:
#         print(f"\nProcessing file....: {file_name}")

#         file_path = os.path.join(train_dir, file_name)

#         try:
#             patches, coords, metadata = generate_patches_from_mat(
#                 file_path, k_top=10, k_mid=10, k_rand=10, patch_size=patch_size
#             )

#             num_patches = len(patches)
#             total_patches += num_patches
#             print(f"Extracted {num_patches} patches from {file_name}")

#             all_patches.extend(patches)
#             all_metadata.extend(metadata)

#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")

#     print(f"\nTotal patches extracted from all files: {total_patches}")

#     if total_patches > 0:
#         idx = 0
#         patch = all_patches[idx]
#         file, x, y, j = all_metadata[idx]
#         visualize_patch(train_dir, patch, file, x, y, j, patch_size)
#     else:
#         print("No patches were extracted.")

if __name__ == "__main__":
    # train_dir = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/trainset"
    train_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/trainset"
    # ADD THIS LINE - specify where to save patches
    # patch_output_dir = "/Users/muhtasimishmumkhan/Desktop/499/hsi/hybArch/HSI_denoising/saved_patches"
    patch_output_dir = "/home/habib/Documents/workspace/hsi_enoising_hybrid/HSI_denoising/saved_patches"
    patch_size = 32

    # ADD THIS LINE - create storage instance
    patch_storage = HSIPatchStorage(patch_output_dir)

    total_patches = 0
    all_patches = []
    all_metadata = []

    mat_files = [f for f in os.listdir(train_dir) if f.endswith(".mat")]

    for file_name in mat_files:
        print(f"\nProcessing file....: {file_name}")

        file_path = os.path.join(train_dir, file_name)

        try:
            patches, coords, metadata = generate_patches_from_mat(
                file_path, k_top=10, k_mid=10, k_rand=10, patch_size=patch_size
            )

            num_patches = len(patches)
            total_patches += num_patches
            print(f"Extracted {num_patches} patches from {file_name}")

            # ADD THESE LINES - save patches for this file immediately
            patch_storage.save_patches_as_mat(patches, metadata, file_name)

            all_patches.extend(patches)
            all_metadata.extend(metadata)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"\nTotal patches extracted from all files: {total_patches}")
    print(f"All patches saved to: {patch_output_dir}")

    if total_patches > 0:
        idx = 0
        patch = all_patches[idx]
        file, x, y, j = all_metadata[idx]
        visualize_patch(train_dir, patch, file, x, y, j, patch_size)
    else:
        print("No patches were extracted.")