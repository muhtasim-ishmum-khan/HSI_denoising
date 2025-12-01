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

    
class test_vis:
    def test_model_comprehensive(model, test_samples, device, patch_size=64):
        """Comprehensive model testing with proper patch handling - SILENT VERSION"""
        model.eval()
        results = []
    
        print("\n" + "="*60)
        print("TESTING MODEL ON ALL SAMPLES")
        print("="*60)
    
        with torch.no_grad():
            for i, sample in enumerate(tqdm(test_samples, desc="Testing samples", ncols=80)):
                try:
                    clean_np = sample['clean']
                    noisy_np = sample['noisy']
    
                    # Create 4D tensors (B, D, H, W)
                    clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).float().to(device)
                    noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).float().to(device)
    
                    B, D, H, W = clean_tensor.shape
    
                    # Process based on image size
                    if H <= patch_size and W <= patch_size:
                        with torch.cuda.amp.autocast():
                            output_tensor = model(noisy_tensor)
                    else:
                        # Patch-based processing for large images
                        output_tensor = torch.zeros_like(clean_tensor)
                        
                        for h_start in range(0, H, patch_size):
                            for w_start in range(0, W, patch_size):
                                h_end = min(h_start + patch_size, H)
                                w_end = min(w_start + patch_size, W)
                                
                                patch_noisy = noisy_tensor[:, :, h_start:h_end, w_start:w_end]
                                
                                # Pad to patch_size if needed
                                pad_h = patch_size - (h_end - h_start)
                                pad_w = patch_size - (w_end - w_start)
                                if pad_h > 0 or pad_w > 0:
                                    patch_noisy = F.pad(patch_noisy, (0, pad_w, 0, pad_h))
                                
                                with torch.cuda.amp.autocast():
                                    patch_output = model(patch_noisy)
                                
                                # Remove padding
                                if pad_h > 0 or pad_w > 0:
                                    patch_output = patch_output[:, :, :h_end-h_start, :w_end-w_start]
                                
                                output_tensor[:, :, h_start:h_end, w_start:w_end] = patch_output
    
                    output_np = output_tensor[0].cpu().numpy()
    
                    # Calculate metrics
                    loss = torch.mean((output_tensor - clean_tensor) ** 2).item()
                    psnr = calculate_psnr(output_tensor, clean_tensor)
                    ssim = calculate_ssim(output_np, clean_np)
                    sam = calculate_sam(output_np, clean_np)
                    
                    input_psnr = calculate_psnr(noisy_tensor, clean_tensor)
                    input_ssim = calculate_ssim(noisy_np, clean_np)
                    input_sam = calculate_sam(noisy_np, clean_np)
    
                    results.append({
                        'filename': sample['filename'],
                        'noise_level': sample['noise_level'],
                        'shape': sample['shape'],
                        'input_psnr': input_psnr,
                        'output_psnr': psnr,
                        'psnr_improvement': psnr - input_psnr,
                        'input_ssim': input_ssim,
                        'ssim': ssim,
                        'ssim_improvement': ssim - input_ssim,
                        'input_sam': input_sam,
                        'sam': sam,
                        'sam_improvement': input_sam - sam,     
                        'loss': loss,
                        'clean': clean_np,
                        'noisy': noisy_np,
                        'denoised': output_np,
                        'original_range': sample['original_range']
                    })
    
                    torch.cuda.empty_cache()
    
                except Exception as e:
                    print(f"\nError processing sample {i+1} ({sample['filename']}): {e}")
                    continue
    
        # Print summary statistics
        if results:
            print(f"\n{'='*60}")
            print(f"TESTING COMPLETED - AVERAGE SCORES")
            print(f"{'='*60}")
            print(f"Total samples tested: {len(results)}/{len(test_samples)}")
            print(f"\nAverage Metrics:")
            print(f"  • Loss:            {np.mean([r['loss'] for r in results]):.6f} ± {np.std([r['loss'] for r in results]):.6f}")
            print(f"  • Input PSNR:      {np.mean([r['input_psnr'] for r in results]):.2f} ± {np.std([r['input_psnr'] for r in results]):.2f} dB")
            print(f"  • Output PSNR:     {np.mean([r['output_psnr'] for r in results]):.2f} ± {np.std([r['output_psnr'] for r in results]):.2f} dB")
            print(f"  • PSNR Improvement: {np.mean([r['psnr_improvement'] for r in results]):.2f} ± {np.std([r['psnr_improvement'] for r in results]):.2f} dB")
            print(f"  • Input SSIM:            {np.mean([r['input_ssim'] for r in results]):.4f} ± {np.std([r['input_ssim'] for r in results]):.4f}")
            print(f"  • Output SSIM:            {np.mean([r['ssim_improvement'] for r in results]):.4f} ± {np.std([r['ssim_improvement'] for r in results]):.4f}")
            print(f"  • SSIM Improvement:            {np.mean([r['ssim'] for r in results]):.4f} ± {np.std([r['ssim'] for r in results]):.4f}")
            print(f"  • Input SAM:             {np.mean([r['input_sam'] for r in results]):.4f} ± {np.std([r['input_sam'] for r in results]):.4f}")
            print(f"  • Output SAM:             {np.mean([r['sam'] for r in results]):.4f} ± {np.std([r['sam'] for r in results]):.4f}")
            print(f"  • SAM Improvement:             {np.mean([r['sam_improvement'] for r in results]):.4f} ± {np.std([r['sam_improvement'] for r in results]):.4f}")
            
            # Best and worst cases
            sorted_by_psnr = sorted(results, key=lambda x: x['output_psnr'])
            print(f"\nPerformance Range:")
            print(f"  • Best PSNR:  {sorted_by_psnr[-1]['output_psnr']:.2f} dB ({sorted_by_psnr[-1]['filename']})")
            print(f"  • Worst PSNR: {sorted_by_psnr[0]['output_psnr']:.2f} dB ({sorted_by_psnr[0]['filename']})")
            print(f"{'='*60}\n")
        else:
            print("\nNo results generated!")
    
        return results
        
    def create_comprehensive_visualizations(results, save_dir):
        """Create comprehensive test result visualizations"""
        print("Creating comprehensive visualizations...")
    
        plt.style.use('default')
        sns.set_palette("husl")
    
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
        noise_level = results[0]['noise_level']
        input_psnrs = [r['input_psnr'] for r in results]
        output_psnrs = [r['output_psnr'] for r in results]
        psnr_improvements = [r['psnr_improvement'] for r in results]
        ssims = [r['ssim'] for r in results]
        sams = [r['sam'] for r in results]
    
        # PSNR comparison
        axes[0, 0].scatter(input_psnrs, output_psnrs, alpha=0.7, s=60, color='blue')
        axes[0, 0].plot([min(input_psnrs), max(output_psnrs)], [min(input_psnrs), max(output_psnrs)], 'r--', alpha=0.5)
        axes[0, 0].axhline(y=40, color='red', linestyle='-', alpha=0.7, label='Target (40 dB)')
        axes[0, 0].set_xlabel('Input PSNR (dB)')
        axes[0, 0].set_ylabel('Output PSNR (dB)')
        axes[0, 0].set_title(f'PSNR: Input vs Output (σ={noise_level})')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
        # PSNR improvement distribution
        axes[0, 1].hist(psnr_improvements, bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(psnr_improvements), color='red', linestyle='--',
                          label=f'Mean: {np.mean(psnr_improvements):.2f} dB')
        axes[0, 1].set_xlabel('PSNR Improvement (dB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('PSNR Improvement Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
        # SSIM distribution
        axes[0, 2].hist(ssims, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(x=np.mean(ssims), color='red', linestyle='--',
                          label=f'Mean: {np.mean(ssims):.3f}')
        axes[0, 2].set_xlabel('SSIM')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('SSIM Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
    
        # SAM distribution
        axes[1, 0].hist(sams, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(sams), color='red', linestyle='--',
                          label=f'Mean: {np.mean(sams):.3f}')
        axes[1, 0].set_xlabel('SAM (radians)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('SAM Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
        # Performance vs Image Size
        image_sizes = [r['shape'][1] * r['shape'][2] for r in results]
        axes[1, 1].scatter(image_sizes, output_psnrs, alpha=0.7, s=60, color='purple')
        axes[1, 1].set_xlabel('Image Size (pixels)')
        axes[1, 1].set_ylabel('Output PSNR (dB)')
        axes[1, 1].set_title('Performance vs Image Size')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Target (40 dB)')
        axes[1, 1].legend()
    
        # Summary statistics
        target_achieved = sum(1 for p in output_psnrs if p > 40)
        success_rate = target_achieved / len(output_psnrs) * 100
        axes[1, 2].axis('off')
        stats_text = f"""
    TEST RESULTS SUMMARY
    {'='*25}
    Static Noise Testing (σ={noise_level})
    Total Samples: {len(results)}
    Unique Images: {len(set([r['filename'] for r in results]))}
    
    PSNR STATISTICS:
    - Mean Output PSNR: {np.mean(output_psnrs):.2f} ± {np.std(output_psnrs):.2f} dB
    - Mean Input PSNR: {np.mean(input_psnrs):.2f} ± {np.std(input_psnrs):.2f} dB
    - Mean Improvement: {np.mean(psnr_improvements):.2f} ± {np.std(psnr_improvements):.2f} dB
    - Target Achievement: {target_achieved} / {len(output_psnrs)} ({success_rate:.1f}%)
    - Max PSNR: {max(output_psnrs):.2f} dB
    - Min PSNR: {min(output_psnrs):.2f} dB
    
    OTHER METRICS:
    - Mean SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}
    - Mean SAM: {np.mean(sams):.4f} ± {np.std(sams):.4f}
    
    CONSISTENCY:
    - PSNR Std Dev: {np.std(output_psnrs):.2f} dB
    - Performance: {'Excellent' if success_rate > 80 else 'Good' if success_rate > 50 else 'Needs Improvement'}
        """
    
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
        plt.suptitle(f'Comprehensive HSI Denoising Test Results (Static noise σ={noise_level})',
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(save_dir, 'comprehensive_test_results.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
        return {
            'mean_psnr': np.mean(output_psnrs),
            'std_psnr': np.std(output_psnrs),
            'mean_improvement': np.mean(psnr_improvements),
            'std_improvement': np.std(psnr_improvements),
            'mean_ssim': np.mean(ssims),
            'std_ssim': np.std(ssims),
            'mean_sam': np.mean(sams),
            'std_sam': np.std(sams),
            'mean_loss': np.mean([r['loss'] for r in results]),
            'std_loss': np.std([r['loss'] for r in results]),
            'target_achievement_rate': success_rate / 100,
            'max_psnr': max(output_psnrs),
            'min_psnr': min(output_psnrs),
            'noise_level': noise_level
        }
    
    def create_sample_visualization(result, title, save_dir):
        """Create detailed visualization with dataset-specific RGB band selection"""
        clean = result['clean']
        noisy = result['noisy'] 
        denoised = result['denoised']
        D, H, W = clean.shape
    
        # Dataset-specific RGB band selection
        filename_lower = result['filename'].lower()
        
        print(f"Creating visualization for {result['filename']}")
        
        # FIXED: Correct RGB band selections for common datasets
        if 'indian' in filename_lower:
            # Indian Pines (220 bands): Standard RGB approximation
            if D >= 150:
                rgb_bands = [49, 26, 16]  # Red: ~630nm, Green: ~550nm, Blue: ~470nm
                print(f"  Using Indian Pines standard RGB bands: {[b+1 for b in rgb_bands]}")
            elif D >= 100:
                rgb_bands = [int(D*0.22), int(D*0.12), int(D*0.07)]
                print(f"  Using scaled Indian bands: {[b+1 for b in rgb_bands]}")
            else:
                rgb_bands = [min(D-1, 22), min(D-1, 12), min(D-1, 7)]
                print(f"  Using minimal Indian bands: {[b+1 for b in rgb_bands]}")
        
        elif 'pavia' in filename_lower:
            # Pavia University/Centre (103 bands)
            if D >= 80:
                rgb_bands = [55, 41, 12]
            else:
                rgb_bands = [int(D*0.7), int(D*0.5), int(D*0.15)]
            print(f"  Using Pavia RGB bands: {[b+1 for b in rgb_bands]}")
        
        elif 'washington' in filename_lower or 'dc' in filename_lower:
            # Washington DC Mall (191 bands)
            if D >= 150:
                rgb_bands = [56, 26, 16]
            else:
                rgb_bands = [int(D*0.35), int(D*0.15), int(D*0.08)]
            print(f"  Using Washington RGB bands: {[b+1 for b in rgb_bands]}")
        
        else:
            # Generic approach: use spread across spectrum
            rgb_bands = [
                min(D-1, int(D*0.7)),   # Near-infrared/Red
                min(D-1, int(D*0.4)),   # Green/Yellow
                min(D-1, int(D*0.15))   # Blue
            ]
            print(f"  Using generic RGB bands: {[b+1 for b in rgb_bands]}")
    
        # Ensure bands are valid
        # rgb_bands = [min(max(0, b), D-1) for b in rgb_bands]
        rgb_bands = [22,13,5]
        
        # Create RGB composites with enhanced contrast
        rgb_images = {}
        for data_type, data in [('clean', clean), ('noisy', noisy), ('denoised', denoised)]:
            rgb_image = np.zeros((H, W, 3))
            for i, band in enumerate(rgb_bands):
                band_data = data[band]
                
                rgb_image[:, :, i] = np.clip(band_data, 0, 1)
            
            # Slight gamma correction for visual appeal
            rgb_image = np.power(rgb_image, 0.95)
            rgb_images[data_type] = rgb_image
    
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    
        axes[0].imshow(rgb_images['clean'], interpolation='nearest')
        axes[0].set_title(f'Clean (Bands {rgb_bands[0]+1}, {rgb_bands[1]+1}, {rgb_bands[2]+1})')
        axes[0].axis('off')
    
        axes[1].imshow(rgb_images['noisy'], interpolation='nearest')
        axes[1].set_title(f'Noisy (Bands {rgb_bands[0]+1}, {rgb_bands[1]+1}, {rgb_bands[2]+1})')
        axes[1].axis('off')
    
        axes[2].imshow(rgb_images['denoised'], interpolation='nearest')
        axes[2].set_title(f'Denoised (Bands {rgb_bands[0]+1}, {rgb_bands[1]+1}, {rgb_bands[2]+1})')
        axes[2].axis('off')
    
        orig_min, orig_max = result['original_range']
        fig.suptitle(f'{title}\n'
                    f'File: {result["filename"]}, Noise: σ={result["noise_level"]:.2f}, '
                    f'Original Range: [{orig_min:.2f}, {orig_max:.2f}]\n'
                    f'PSNR: {result["input_psnr"]:.2f}→{result["output_psnr"]:.2f} dB (+{result["psnr_improvement"]:.2f}), '
                    f'SSIM: {result["ssim"]:.3f}, SAM: {result["sam"]:.3f}',
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        safe_title = title.replace(' ', '_').lower()
        safe_filename = result['filename'].replace('.mat', '').replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f'sample_{safe_title}_{safe_filename}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"  Saved visualization: sample_{safe_title}_{safe_filename}.png")
    
    def create_detailed_analysis_plots(results, save_dir):
        """Create detailed analysis plots"""
        print("Creating detailed analysis plots...")
    
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
        input_psnrs = [r['input_psnr'] for r in results]
        output_psnrs = [r['output_psnr'] for r in results]
        psnr_improvements = [r['psnr_improvement'] for r in results]
        ssims = [r['ssim'] for r in results]
        sams = [r['sam'] for r in results]
        losses = [r['loss'] for r in results]
        noise_level = results[0]['noise_level']
        image_sizes = [r['shape'][1] * r['shape'][2] for r in results]
    
        # Correlation matrix
        try:
            import pandas as pd
            metrics_data = {
                'Input_PSNR': input_psnrs,
                'Output_PSNR': output_psnrs,
                'PSNR_Improvement': psnr_improvements,
                'SSIM': ssims,
                'SAM': sams,
                'Loss': losses,
                'Image_Size': image_sizes
            }
            df = pd.DataFrame(metrics_data)
            correlation_matrix = df.corr()
    
            im = axes[0, 0].imshow(correlation_matrix.values, cmap='RdYlBu', vmin=-1, vmax=1)
            axes[0, 0].set_xticks(range(len(correlation_matrix.columns)))
            axes[0, 0].set_yticks(range(len(correlation_matrix.columns)))
            axes[0, 0].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            axes[0, 0].set_yticklabels(correlation_matrix.columns)
            axes[0, 0].set_title('Metrics Correlation Matrix')
    
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    axes[0, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
            plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        except ImportError:
            axes[0, 0].text(0.5, 0.5, 'pandas not available\nfor correlation matrix',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
    
        # PSNR improvement distribution
        axes[0, 1].hist(psnr_improvements, bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(psnr_improvements), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(psnr_improvements):.2f} dB')
        axes[0, 1].axvline(x=np.median(psnr_improvements), color='blue', linestyle='--', linewidth=2,
                          label=f'Median: {np.median(psnr_improvements):.2f} dB')
        axes[0, 1].set_xlabel('PSNR Improvement (dB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'PSNR Improvement Distribution (σ={noise_level})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
        # Performance vs Image Size
        scatter = axes[1, 0].scatter(image_sizes, output_psnrs, alpha=0.7, s=60,
                                    c=psnr_improvements, cmap='RdYlGn')
        axes[1, 0].set_xlabel('Image Size (pixels)')
        axes[1, 0].set_ylabel('Output PSNR (dB)')
        axes[1, 0].set_title('PSNR vs Image Size (colored by improvement)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Target (40 dB)')
        axes[1, 0].legend()
        plt.colorbar(scatter, ax=axes[1, 0], label='PSNR Improvement (dB)', shrink=0.8)
    
        # Performance summary
        target_achieved = sum(1 for p in output_psnrs if p > 40)
        success_rate = target_achieved / len(output_psnrs) * 100
        consistency_metric = np.std(psnr_improvements) / np.mean(psnr_improvements) if np.mean(psnr_improvements) > 0 else 0
        stats_text = f"""
    DETAILED PERFORMANCE ANALYSIS
    {'='*35}
    Static Noise Testing (σ={noise_level})
    
    PSNR PERFORMANCE:
    - Mean Output PSNR: {np.mean(output_psnrs):.2f} ± {np.std(output_psnrs):.2f} dB
    - Mean Input PSNR: {np.mean(input_psnrs):.2f} ± {np.std(input_psnrs):.2f} dB
    - Mean Improvement: {np.mean(psnr_improvements):.2f} ± {np.std(psnr_improvements):.2f} dB
    - Max Improvement: {max(psnr_improvements):.2f} dB
    - Min Improvement: {min(psnr_improvements):.2f} dB
    
    TARGET ACHIEVEMENT:
    - Success Rate: {success_rate:.1f}% ({target_achieved}/{len(results)})
    - Above 40 dB: {target_achieved} samples
    - Above 35 dB: {sum(1 for p in output_psnrs if p > 35)} samples
    - Above 30 dB: {sum(1 for p in output_psnrs if p > 30)} samples
    
    CONSISTENCY ANALYSIS:
    - Coefficient of Variation: {consistency_metric:.3f}
    - Performance Stability: {'High' if consistency_metric < 0.2 else 'Moderate' if consistency_metric < 0.5 else 'Low'}
    
    QUALITY METRICS:
    - SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}
    - SAM: {np.mean(sams):.4f} ± {np.std(sams):.4f}
    
    IMAGE SIZE ANALYSIS:
    - Min Size: {min(image_sizes)} pixels
    - Max Size: {max(image_sizes)} pixels
    - Mean Size: {np.mean(image_sizes):.0f} pixels
        """
    
        axes[1, 1].axis('off')
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
        plt.suptitle(f'Detailed Analysis - Static Noise Level (σ={noise_level})',
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(save_dir, 'detailed_analysis_plots.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
        return {
            'mean_psnr': np.mean(output_psnrs),
            'std_psnr': np.std(output_psnrs),
            'mean_improvement': np.mean(psnr_improvements),
            'std_improvement': np.std(psnr_improvements),
            'mean_ssim': np.mean(ssims),
            'std_ssim': np.std(ssims),
            'mean_sam': np.mean(sams),
            'std_sam': np.std(sams),
            'mean_loss': np.mean([r['loss'] for r in results]),
            'std_loss': np.std([r['loss'] for r in results]),
            'target_achievement_rate': success_rate / 100,
            'max_psnr': max(output_psnrs),
            'min_psnr': min(output_psnrs),
            'noise_level': noise_level
        }
    
    def create_spectral_analysis(results, save_dir, num_samples=3):
        """Create spectral signature analysis"""
        print("Creating spectral analysis...")
    
        sorted_results = sorted(results, key=lambda x: x['output_psnr'])
        sample_indices = [0, len(sorted_results)//2, len(sorted_results)-1]
        selected_results = [sorted_results[i] for i in sample_indices]
    
        fig, axes = plt.subplots(len(selected_results), 1, figsize=(15, 4*len(selected_results)))
        if len(selected_results) == 1:
            axes = [axes]
    
        for idx, result in enumerate(selected_results):
            clean = result['clean']
            noisy = result['noisy']
            denoised = result['denoised']
    
            D, H, W = clean.shape
            center_h, center_w = H//2, W//2
    
            clean_spectrum = clean[:, center_h, center_w]
            noisy_spectrum = noisy[:, center_h, center_w]
            denoised_spectrum = denoised[:, center_h, center_w]
    
            bands = np.arange(D)
    
            axes[idx].plot(bands, clean_spectrum, 'g-', linewidth=2, label='Clean', alpha=0.8)
            axes[idx].plot(bands, noisy_spectrum, 'r--', linewidth=1.5, label='Noisy', alpha=0.7)
            axes[idx].plot(bands, denoised_spectrum, 'b-', linewidth=2, label='Denoised', alpha=0.8)
    
            axes[idx].set_xlabel('Spectral Band')
            axes[idx].set_ylabel('Reflectance')
            axes[idx].set_title(f'Spectral Signature - {result["filename"]}\n'
                               f'PSNR: {result["output_psnr"]:.2f} dB, SSIM: {result["ssim"]:.3f}, '
                               f'SAM: {result["sam"]:.4f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
            ax2 = axes[idx].twinx()
            denoising_improvement = np.abs(denoised_spectrum - clean_spectrum) - np.abs(noisy_spectrum - clean_spectrum)
            ax2.fill_between(bands, 0, denoising_improvement, alpha=0.3, color='purple', label='Denoising Effect')
            ax2.set_ylabel('Denoising Effect', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
    
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spectral_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(results, summary_stats, save_dir, model_info=None):
        """Save detailed results as visualizations"""
        print("Saving visualization results...")
    
        print(f"Results saved to {save_dir}")
        print(f"• Comprehensive plots: comprehensive_test_results.png")
        print(f"• Detailed analysis: detailed_analysis_plots.png")
        print(f"• Spectral analysis: spectral_analysis.png")
        print(f"• Sample visualizations: sample_*.png")
        print(f"• Testing Mode: Static noise (σ={summary_stats['noise_level']})")
