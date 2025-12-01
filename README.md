# CASync: Cross-Attention for Single-Speaker Lip Sync

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

**A lightweight, efficient, and high-fidelity lip synchronization model for single-speaker finetuning.**

[Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Performance](#performance) • [Comparison](#qualitative-comparison)

</div>

---

## 📖 Introduction

**CASync** is a novel audio-driven lip synchronization framework designed for single-speaker scenarios. Unlike traditional methods that rely on heavy external discriminators (e.g., SyncNet) or massive datasets, CASync leverages a **cross-attention mechanism** to implicitly learn precise audio-visual alignment.

Our approach achieves superior lip-sync accuracy and visual quality with a significantly smaller model size (19.8M params) and lower computational cost (4.08 GFLOPs), making it ideal for real-time applications and resource-constrained environments.

## ✨ Key Features

- **⚡ Lightweight & Efficient**: Only **19.8M parameters** and **4.08 GFLOPs**, enabling real-time inference on consumer GPUs.
- **🎯 Precise Lip Sync**: Uses **cross-attention** to align audio and visual features without explicit SyncNet supervision.
- **👤 Single-Speaker Finetuning**: Requires only **1-5 minutes** of training data for a specific target identity.
- **🖼️ High Fidelity**: Preserves fine facial details (e.g., moles, beard texture) and identity characteristics better than GAN-based or inpainting-based methods.
- **🚀 End-to-End Training**: Simple supervised reconstruction objective (L1 + Perceptual Loss), avoiding unstable adversarial training.

## 📊 Performance

We compare CASync with state-of-the-art methods in terms of model efficiency and visual quality.

### Model Efficiency

| Model | Parameters (M) | FLOPs (G) |
|-------|---------------:|----------:|
| **CASync (Ours)** | **19.79** | **4.08** |
| Wav2Lip | 36.30 | 3.99 |
| VideoRetalking | 325.09 | 174.80 |
| MuseTalk | 946.90 | 548.40 |
| LatentSync | 1,360.00 | 6,230.00 |

![Model Comparison](model_comparison.png)

### Qualitative Comparison

Comparison with SOTA methods in the cross-generation setting. CASync demonstrates superior clarity and identity preservation.

![Qualitative Results](comparison_figure.jpg)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CASync.git
   cd CASync
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers opencv-python librosa soundfile scipy tqdm matplotlib
   
   # For FLOPs counting (optional)
   pip install thop
   ```

3. **Install FFmpeg**
   - **Ubuntu**: `sudo apt install ffmpeg`
   - **MacOS**: `brew install ffmpeg`
   - **Windows**: Download and add to PATH.

## 🚀 Quick Start

### 1. Data Preprocessing
Extract audio features (HuBERT) and facial landmarks from your video.

```bash
# Process a single video file
python step1_data_preprocess.py
```
*Note: Edit the script to specify your video path and output directory.*

### 2. Training
Train the model on the processed data.

```bash
python step2_train_unet.py
```
*Note: Edit the script to set dataset paths and hyperparameters.*

### 3. Inference
Generate a lip-synced video using a trained model and a driving audio.

```bash
python inference.py
```
*Note: Edit the script to specify model path, source video, and driving audio.*

## 📂 Project Structure

```
CASync/
├── data/                   # Data storage
├── dataset/                # Dataset loading logic
├── image_infer_v1/         # Inference utilities
├── module/                 # Model architecture (UNet, Attention)
│   └── unet.py             # Main model file
├── pretrained_models/      # Pretrained weights (HuBERT, VGG, etc.)
├── utils/                  # Helper functions
│   ├── hubert_extractor.py # Audio feature extraction
│   └── lip_detector/       # Face landmark detection
├── step1_data_preprocess.py # Data preprocessing script
├── step2_train_unet.py     # Training script
├── inference.py            # Inference script
├── debug_unet.py           # Model analysis tool (Params/FLOPs)
├── create_comparison_figure.py # Visualization tool
└── README.md
```

## 🧩 Implementation Details

- **Inputs**: 
  - **Visual**: Single cropped mouth region (160x160) + Mask.
  - **Audio**: HuBERT features (resampled to 16kHz, windowed).
- **Architecture**:
  - **FaceEncoder**: Lightweight U-Net encoder (Depthwise Separable Convs).
  - **AudioEncoder**: HuBERT backbone + 7-layer CNN.
  - **Fusion**: MLP + 4 Cross-Attention Blocks.
  - **Decoder**: U-Net decoder with skip connections.
- **Loss Function**: L1 Reconstruction Loss + 0.1 * VGG Perceptual Loss.

## 📝 Citation

If you find this project useful, please cite our paper:

```bibtex
@article{casync2025,
  title={CASync: Cross-Attention for Single-Speaker Lip Sync},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 Acknowledgements

We acknowledge the following open-source projects:
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- [Face Alignment](https://github.com/1adrianb/face-alignment)

---
<div align="center">
Created with ❤️ by the CASync Team
</div>

#
