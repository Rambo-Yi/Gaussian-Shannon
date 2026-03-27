# Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication

Official PyTorch implementation of the paper: **"Gaussian Shannon"**, accepted at CVPR 2026 Findings.

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026%20Findings-blue.svg" alt="Conference">
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D1.10-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>


## 🎨 Overview

![framework](https://github.com/user-attachments/assets/5b3dbda5-737f-4e70-8377-0c4e8a4c0e3f)

This study aims to construct a high-precision watermarking framework for diffusion models based on a communication mechanism. The generation process of diffusion models, DDIM Inversion, and external image attacks are uniformly modeled as a noisy communication process. A architecture is designed that balances "robust traceability" and "lossless information recovery." On one hand, redundancy design ensures identity recognition under strong attacks; on the other hand, precise decoding enables bit-exact recovery of metadata. Specifically, majority voting leverages spatial redundancy to combat severe local distortions in latent-space images, while error-correcting codes correct residual random errors, significantly improving the accuracy of watermark recovery.

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/Rambo-Yi/Gaussian-Shannon.git
cd Gaussian-Shannon

# Setup environment
conda create -n gs_env python=3.12 -y
conda activate gs_env
pip install -r requirements.txt
```
