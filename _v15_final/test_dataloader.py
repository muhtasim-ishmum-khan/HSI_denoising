import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import warnings
from einops import rearrange
warnings.filterwarnings('ignore')
import h5py
try:
    import scipy.io as sio
except ImportError:
    sio = None

    
class test_dataloader:
    def diagnose_dataset_structure(file_path):
        """Diagnose the structure of a .mat file to understand data organization"""
        if not sio:
            print("scipy not available")
            return None
            
        try:
            mat = sio.loadmat(file_path)
            print(f"\n=== Diagnosing: {os.path.basename(file_path)} ===")
            
            # Print all keys
            keys = [k for k in mat.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            
            for key in keys:
                data = mat[key]
                print(f"Key '{key}': shape={data.shape}, dtype={data.dtype}")
                print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
                print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
                
                # Check if it looks like spectral data
                if len(data.shape) == 3:
                    print(f"  3D data - possible interpretations:")
                    print(f"    As (H, W, Bands): {data.shape}")
                    print(f"    As (Bands, H, W): {data.shape}")
                    
                    # Check which dimension might be spectral
                    dim_variances = [data.var(axis=i).mean() for i in range(3)]
                    spectral_dim = np.argmax(dim_variances)
                    print(f"  Dimension variances: {dim_variances}")
                    print(f"  Likely spectral dimension: {spectral_dim} (highest variance)")
                    
            return mat, keys
        except Exception as e:
            print(f"Error diagnosing {file_path}: {e}")
            return None, []
            
    def load_test_data(test_dir, target_bands=31, max_files=None, test_crop_size=512):
        """Load test data with ICVL-specific preprocessing: 512×512×31 crops"""
        print(f"Loading test data from: {test_dir}")
    
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return []
    
        mat_files = glob.glob(os.path.join(test_dir, '*.mat'))
        if max_files:
            mat_files = mat_files[:max_files]
    
        print(f"Found {len(mat_files)} test files\n")
    
        test_data = []
        failed_files = []
        
        for file_path in tqdm(mat_files, desc="Loading test files", ncols=80):
            try:
                # Try loading with h5py first
                with h5py.File(file_path, 'r') as f:
                    if 'rad' in f:
                        data_key = 'rad'
                        cube_raw = np.array(f[data_key]).astype(np.float32)
                    else:
                        keys = [k for k in f.keys() if not k.startswith('#')]
                        if keys:
                            data_key = keys[0]
                            cube_raw = np.array(f[data_key]).astype(np.float32)
                        else:
                            failed_files.append((os.path.basename(file_path), "No valid keys found"))
                            continue
                            
            except Exception as h5_error:
                if sio:
                    try:
                        mat = sio.loadmat(file_path)
                        keys = [k for k in mat.keys() if not k.startswith('__')]
                        data_key = keys[0] if keys else None
                        if data_key:
                            cube_raw = mat[data_key].astype(np.float32)
                        else:
                            failed_files.append((os.path.basename(file_path), "No valid keys in scipy load"))
                            continue
                    except Exception as scipy_error:
                        failed_files.append((os.path.basename(file_path), f"Both loaders failed"))
                        continue
                else:
                    failed_files.append((os.path.basename(file_path), "h5py failed, scipy unavailable"))
                    continue
    
            try:
                cube_original = cube_raw.copy()
    
                # ICVL format handling: (D, H, W) -> transpose to (H, W, D)
                if cube_raw.ndim == 3:
                    if cube_raw.shape[0] < min(cube_raw.shape[1:]):  # (D, H, W)
                        cube_raw = cube_raw.transpose(1, 2, 0)  # -> (H, W, D)
                        cube_original = cube_original.transpose(1, 2, 0)
                elif cube_raw.ndim == 2:
                    cube_raw = cube_raw[np.newaxis, np.newaxis, ...]
                    cube_original = cube_original[np.newaxis, np.newaxis, ...]
    
                if cube_raw.ndim != 3:
                    failed_files.append((os.path.basename(file_path), f"Invalid dimensions: {cube_raw.ndim}"))
                    continue
    
                H, W, D = cube_raw.shape
    
                # CRITICAL: Center crop to 512×512 for testing (ICVL protocol)
                if H > test_crop_size or W > test_crop_size:
                    start_h = (H - test_crop_size) // 2
                    start_w = (W - test_crop_size) // 2
                    cube_raw = cube_raw[start_h:start_h+test_crop_size,
                                       start_w:start_w+test_crop_size, :]
                    cube_original = cube_original[start_h:start_h+test_crop_size,
                                                 start_w:start_w+test_crop_size, :]
                    H, W = test_crop_size, test_crop_size
                    #print(f"  Cropped {os.path.basename(file_path)} to {test_crop_size}×{test_crop_size}")
                elif H < test_crop_size or W < test_crop_size:
                    # Pad if smaller than 512
                    pad_h = max(0, test_crop_size - H)
                    pad_w = max(0, test_crop_size - W)
                    cube_raw = np.pad(cube_raw, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                    cube_original = np.pad(cube_original, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                    H, W = test_crop_size, test_crop_size
                    print(f"  Padded {os.path.basename(file_path)} to {test_crop_size}×{test_crop_size}")
    
                # Select exactly 31 bands (ICVL standard)
                if D >= target_bands:
                    cube_raw = cube_raw[:, :, :target_bands]
                    cube_original = cube_original[:, :, :target_bands]
                else:
                    failed_files.append((os.path.basename(file_path), f"Insufficient bands: {D} < {target_bands}"))
                    continue
    
                # Store original range
                original_min = cube_original.min()
                original_max = cube_original.max()
    
                # MATCH TRAINING: Global min-max normalization to [0,1]
                cube = (cube_raw - cube_raw.min()) / (cube_raw.max() - cube_raw.min() + 1e-8)
    
                # Convert to (D, H, W) format for model
                cube = cube.transpose(2, 0, 1)
    
                test_data.append({
                    'clean': cube,
                    'original_range': (original_min, original_max),
                    'filename': os.path.basename(file_path),
                    'shape': cube.shape,
                    'dataset_type': 'icvl'
                })
                
            except Exception as e:
                failed_files.append((os.path.basename(file_path), f"Processing error: {str(e)[:50]}"))
                continue
    
        # Summary
        print(f"\n{'='*60}")
        print(f"DATA LOADING SUMMARY (ICVL Protocol: {test_crop_size}×{test_crop_size}×{target_bands})")
        print(f"{'='*60}")
        print(f"✓ Successfully loaded: {len(test_data)}/{len(mat_files)} files")
        
        if failed_files:
            print(f"✗ Failed to load: {len(failed_files)} files")
            for fname, reason in failed_files[:5]:
                print(f"    - {fname}: {reason}")
            if len(failed_files) > 5:
                print(f"    ... and {len(failed_files) - 5} more")
        
        print(f"{'='*60}\n")
    
        return test_data
        
    def add_static_noise_to_data(clean_data, noise_level=50, batch_size=5):
        """Add Gaussian noise matching TRAINING protocol exactly"""
        import gc
        
        test_samples = []
        total_samples = len(clean_data)
        num_batches = (total_samples + batch_size - 1) // batch_size
        failed_samples = []
        
        print(f"Adding Gaussian noise (σ={noise_level}) to {total_samples} samples...")
        print(f"Noise protocol: MATCH TRAINING (scale [0,1] -> [0,255], add noise, clip, rescale)")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_samples)
            batch_data = clean_data[batch_start:batch_end]
            
            for local_idx, data_item in enumerate(batch_data):
                global_idx = batch_start + local_idx
                filename = data_item['filename']
                
                try:
                    clean_raw = data_item['clean']
                    
                    if np.isnan(clean_raw).any() or np.isinf(clean_raw).any():
                        raise ValueError(f"Invalid values in {filename}")
    
                    with torch.no_grad():
                        clean_tensor = torch.from_numpy(clean_raw).float().unsqueeze(0)
                        
                        # EXACT MATCH WITH TRAINING: Paper-standard Gaussian noise
                        clean_255 = clean_tensor * 255.0
                        noise = torch.randn_like(clean_255) * noise_level
                        noisy_255 = torch.clamp(clean_255 + noise, 0, 255)
                        noisy_normalized = noisy_255 / 255.0
    
                        clean_np = clean_tensor[0].cpu().numpy()
                        noisy_np = noisy_normalized[0].cpu().numpy()
                        
                        del clean_tensor, clean_255, noise, noisy_255, noisy_normalized
    
                    test_samples.append({
                        'clean': clean_np,
                        'noisy': noisy_np,
                        'noise_level': noise_level / 255.0,
                        'original_range': data_item['original_range'],
                        'filename': filename,
                        'shape': clean_np.shape
                    })
                    
                except Exception as e:
                    failed_samples.append((global_idx + 1, filename, str(e)))
                    continue
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        print(f"✓ Successfully processed: {len(test_samples)}/{total_samples} samples\n")
        
        return test_samples
