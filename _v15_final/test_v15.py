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

from test_dataloader import test_dataloader
from test_vis import test_vis
from metrics_hsi import metrics_hsi
from LR_scheduler import WarmupCosineScheduler

def main():
    """Main testing function with ICVL protocol"""
    print("="*80)
    print("HSI DENOISING MODEL TESTING - ICVL Protocol")
    print("Test crops: 512×512×31 with training-matched normalization")
    print("="*80)

    # Configuration --- Change directories accordingly
    
    TEST_DIR = '/workspace/icvl_part/test_gauss'
    MODEL_PATH = './HSI_denoising_ICVL_resultsV15_noise30'
    RESULTS_DIR = './HSI_denoising_ICVL_resultsV15_noise30/test_results'
    TRAINING_NOISE_LEVEL = 10 #change according to training noise level
    
    config = {
        'base_dim': 64,
        'target_bands': 31,  # ICVL standard
        'patch_size': 64,    # For patch-based processing if needed
        'test_crop_size': 512,  # ICVL test protocol
        'noise_level': TRAINING_NOISE_LEVEL,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    # Load model
    print("\nLoading trained model...")
    full_model_path = os.path.join(MODEL_PATH, 'enhanced_denoising_pipeline_full.pth')
    best_model_path = os.path.join(MODEL_PATH, 'best_memory_optimized_model.pth')

    model_info = None
    if os.path.exists(full_model_path):
        print("Loading full model with metadata...")
        checkpoint = torch.load(full_model_path, map_location=device, weights_only=False)
        model_info = checkpoint
        config = checkpoint.get('config', config)
    elif os.path.exists(best_model_path):
        print("Loading best model checkpoint...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', config)
    else:
        raise FileNotFoundError(f"No model found in {MODEL_PATH}")

    # Initialize model
    model = MemoryOptimizedUNet(
        in_channels=1,
        base_dim=config['base_dim'],
        window_sizes=[4, 8, 16],
        num_bands=config['target_bands']
    ).to(device)

    # Escape strict model parameter matching here
    #model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # ===================================================================
    # FIXED: Handle spectral_pos_embed size mismatches
    # ===================================================================
    # Get the saved state dict and current model's state dict
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Prepare filtered state dict
    filtered_state_dict = {}
    adjusted_keys = []
    skipped_keys = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                # Shapes match - load directly
                filtered_state_dict[k] = v
            else:
                # Size mismatch detected
                if 'spectral_pos_embed' in k:
                    # Handle spectral positional embedding mismatches
                    needed_bands = model_dict[k].shape[1]
                    available_bands = v.shape[1]
                    
                    if needed_bands <= available_bands:
                        # Checkpoint has more bands than we need - slice it
                        filtered_state_dict[k] = v[:, :needed_bands, :].clone()
                        adjusted_keys.append(f"{k}: {v.shape} -> {filtered_state_dict[k].shape}")
                    else:
                        # Checkpoint has fewer bands than we need - pad it
                        padded = torch.zeros_like(model_dict[k])
                        padded[:, :available_bands, :] = v
                        filtered_state_dict[k] = padded
                        adjusted_keys.append(f"{k}: {v.shape} -> {filtered_state_dict[k].shape} (padded)")
                else:
                    # Non-spectral parameter with mismatch - skip it
                    skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
        else:
            # Key not in model - likely from different architecture
            skipped_keys.append(f"{k}: not found in current model")
    
    # Report what was adjusted
    if adjusted_keys:
        print(f"\n✓ Adjusted {len(adjusted_keys)} spectral_pos_embed parameters:")
        for key in adjusted_keys[:5]:  # Show first 5
            print(f"    {key}")
        if len(adjusted_keys) > 5:
            print(f"    ... and {len(adjusted_keys) - 5} more")
    
    if skipped_keys:
        print(f"\n⚠ Skipped {len(skipped_keys)} mismatched parameters:")
        for key in skipped_keys[:3]:  # Show first 3
            print(f"    {key}")
        if len(skipped_keys) > 3:
            print(f"    ... and {len(skipped_keys) - 3} more")
    
    # Load the filtered state dict (strict=False allows missing keys)
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=True)
    
    if missing_keys:
        print(f"\n⚠ Missing keys (will use random initialization): {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    {key}")
    
    if unexpected_keys:
        print(f"\n⚠ Unexpected keys (ignored): {len(unexpected_keys)}")
    
    print("\n✓ Model weights loaded successfully!")
    # ===================================================================
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Parameters: {total_params / 1e6:.2f}M")
    print(f"Base dimension: {config['base_dim']}")
    print(f"Target bands: {config['target_bands']}")
    print(f"Training noise level: σ={config['noise_level']}")

    if model_info and 'best_psnr' in model_info:
        print(f"Training best PSNR: {model_info['best_psnr']:.2f} dB")

    # Load test data
    test_data = load_test_data(
        TEST_DIR, 
        target_bands=config['target_bands'],
        test_crop_size=config.get('test_crop_size', 512)
    )

    if not test_data:
        print("ERROR: No test data loaded!")
        return
        

    print(f"Loaded {len(test_data)} test images (512×512×31)")

    # Add static noise
    test_samples = add_static_noise_to_data(test_data, noise_level=config['noise_level'])

    print("\nNoise Analysis:")
    noise_types = {}
    actual_noise_levels = []
    for sample in test_samples:
        noise_type = sample.get('noise_type', 'unknown')
        noise_types[noise_type] = noise_types.get(noise_type, 0) + 1
        actual_noise_levels.append(sample.get('actual_noise_level', 0))
    
    print(f"Noise types distribution: {noise_types}")
    print(f"Actual noise levels - Mean: {np.mean(actual_noise_levels):.4f}, "
          f"Std: {np.std(actual_noise_levels):.4f}, "
          f"Range: [{np.min(actual_noise_levels):.4f}, {np.max(actual_noise_levels):.4f}]")
    
    # Verify noise is visible
    sample_noise = test_samples[0]['noisy'] - test_samples[0]['clean']
    print(f"Sample noise verification - Mean: {np.mean(sample_noise):.4f}, "
          f"Std: {np.std(sample_noise):.4f}, "
          f"Max: {np.max(np.abs(sample_noise)):.4f}")

    print(f"Created {len(test_samples)} test samples with static noise σ={config['noise_level']}")

    # Test model
    results = test_model_comprehensive(model, test_samples, device, patch_size=config['patch_size'])

    if not results:
        print("No test results generated!")
        return

    print(f"\nSuccessfully tested {len(results)} samples")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS:")
    print("="*60)
    print(f"Number of test samples: {len(results)}")
    print(f"Average Test Loss: {np.mean([r['loss'] for r in results]):.6f} ± {np.std([r['loss'] for r in results]):.6f}")
    print(f"Average PSNR:      {np.mean([r['output_psnr'] for r in results]):.3f} ± {np.std([r['output_psnr'] for r in results]):.3f} dB")
    print(f"Average SSIM:      {np.mean([r['ssim'] for r in results]):.4f} ± {np.std([r['ssim'] for r in results]):.4f}")
    print(f"Average SAM:       {np.mean([r['sam'] for r in results]):.4f} ± {np.std([r['sam'] for r in results]):.4f}")

    sorted_results = sorted(results, key=lambda x: x['output_psnr'])
    best_result = sorted_results[-1]
    worst_result = sorted_results[0]
    print(f"Best PSNR:  {best_result['filename']} ({best_result['output_psnr']:.3f} dB)")
    print(f"Worst PSNR: {worst_result['filename']} ({worst_result['output_psnr']:.3f} dB)")
    print("="*60)

    # Visualizations
    summary_stats = create_comprehensive_visualizations(results, RESULTS_DIR)

    sorted_results = sorted(results, key=lambda x: x['output_psnr'])
    best_result = sorted_results[-1]
    worst_result = sorted_results[0]
    median_result = sorted_results[len(sorted_results)//2]

    for title, result in [('Best Performance', best_result),
                         ('Median Performance', median_result),
                         ('Challenging Case', worst_result)]:
        create_sample_visualization(result, title, RESULTS_DIR)

    create_spectral_analysis(results, RESULTS_DIR)
    save_detailed_results(results, summary_stats, RESULTS_DIR, model_info)

    # Final summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*80)

    print(f"Test Configuration:")
    print(f"  • Test Directory: {TEST_DIR}")
    print(f"  • Model Path: {MODEL_PATH}")
    print(f"  • Total Samples: {len(results)}")
    print(f"  • Unique Images: {len(set([r['filename'] for r in results]))}")
    print(f"  • Noise Level: σ={summary_stats['noise_level']}")
    print(f"  • Device: {device}")

    print(f"\nPerformance Metrics:")
    print(f"  • Mean PSNR: {summary_stats['mean_psnr']:.2f} ± {summary_stats['std_psnr']:.2f} dB")
    print(f"  • Mean Improvement: {summary_stats['mean_improvement']:.2f} dB")
    print(f"  • Max PSNR: {summary_stats['max_psnr']:.2f} dB")
    print(f"  • Min PSNR: {summary_stats['min_psnr']:.2f} dB")
    print(f"  • Mean SSIM: {summary_stats['mean_ssim']:.4f} ± {summary_stats['std_ssim']:.4f}")
    print(f"  • Mean SAM: {summary_stats['mean_sam']:.4f} ± {summary_stats['std_sam']:.4f}")

    print(f"\nTarget Achievement (PSNR > 40 dB):")
    achievement_rate = summary_stats['target_achievement_rate'] * 100
    target_count = int(achievement_rate * len(results) / 100)
    print(f"  • Success Rate: {achievement_rate:.1f}% ({target_count} / {len(results)} samples)")
    print(f"  • Status: {'EXCELLENT' if achievement_rate > 80 else 'GOOD' if achievement_rate > 50 else 'NEEDS IMPROVEMENT'}")

    print(f"\nModel Architecture:")
    print(f"  • Parameters: {total_params / 1e6:.2f}M")
    print(f"  • Base Dimension: {config['base_dim']}")
    print(f"  • Input Bands: {config['target_bands']}")
    print(f"  • Patch Size: {config['patch_size']}")

    if model_info and 'best_psnr' in model_info:
        training_psnr = model_info['best_psnr']
        test_psnr = summary_stats['mean_psnr']
        generalization = "Good" if abs(training_psnr - test_psnr) < 5 else "Moderate" if abs(training_psnr - test_psnr) < 10 else "Poor"
        print(f"\nGeneralization Analysis:")
        print(f"  • Training PSNR: {training_psnr:.2f} dB")
        print(f"  • Test PSNR: {test_psnr:.2f} dB")
        print(f"  • Difference: {abs(training_psnr - test_psnr):.2f} dB")
        print(f"  • Generalization: {generalization}")

    print(f"\n" + "="*80)
    print(f"Final Average PSNR: {summary_stats['mean_psnr']:.3f} ± {summary_stats['std_psnr']:.3f} dB")
    print(f"Final Average SSIM: {summary_stats['mean_ssim']:.4f} ± {summary_stats['std_ssim']:.4f}")
    print(f"Final Average SAM:  {summary_stats['mean_sam']:.4f} ± {summary_stats['std_sam']:.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*80)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*80)

    return results, summary_stats, model_info

if __name__ == '__main__':
    main()