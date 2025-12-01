# An Adaptive Spatial-Spectral Model for Denoising Hyperspectral Images
## Abstract
#### Compared to typical RGB images, hyperspectral images (HSIs) have significantly more spectral information, revealing comprehensive properties of materials, enabling applications in remote sensing, agriculture, environmental monitoring, and medical imaging. However, HSI acquisition process often suffers from unwanted noise that severely degrades the acquired HSI quality, limiting widespread usage. Driven by that motivation, we propose a denoising framework combining Spectral-Spatial Transformer with hierarchical U-Net architecture, for joint spatial-spectral processing. Our model achieves 40 db PSNR on ICVL dataset, that is comparable with state-of-the-art methods. By preserving spectral fidelity alongside spatial clarity, our approach enables effective usage of HSIs in downstream applications.

## The architecture
<img width="800" height="400" alt="step2" src="https://github.com/muhtasim-ishmum-khan/muhtasim-ishmum-khan/blob/main/FIG1_arch.png" />

## Results: Model performance over different noise levels
### The Loss curve
<img width="300" height="400" alt="step2" src="Loss.png" />

### Progression of image quality
#### Peak-to-peak Signal Ratio (PSNR)
<img width="300" height="400" alt="step2" src="PSNR.png" />

#### Spectral Angle Mapper (SAM)
<img width="300" height="400" alt="step2" src="SAM.png" />

#### Structural Similiarity Index Measure
<img width="300" height="400" alt="step2" src="SSIM.png" />

## Running the code
### The codes for training and testing are available in _v15_final. The saved models are currently excluded due to size limitations.
### Step 1:
#### Navigate to the files: "train_v15.py" and "test_v15.py" and change the paths of the directories as required.
### Step 2:
#### Use the notebook: run_training_testing.ipynb to run testing or training.


