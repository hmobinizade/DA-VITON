# DA-VITON: Depth-Attention Virtual Try-On

**DA-VITON** is a high-resolution, image-based virtual try-on framework that enhances garment-body alignment, occlusion handling, and detail preservation through depth-aware attention mechanisms. This repository contains the official implementation of the Depth-Attention VTON architecture.

## 🧠 Key Features

- 🔍 **Depth-Aware Garment Refinement**: Removes irrelevant garment regions (e.g., inner collar) using depth maps for clean and precise preprocessing.
- 🧵 **Dual Encoder with Attention**: Separately encodes garment and body features using ResNet-based encoders enriched with multi-head attention layers.
- 🧍‍♂️ **Garment-Body Integration Module**: Fuses depth-informed garment and body features for accurate alignment.
- 🎨 **High-Resolution Image Generator**: Synthesizes realistic try-on images using a pre-trained generator via transfer learning.

## 📦 Dataset & Depth Maps

- **Dataset**: [VITON-HD on Kaggle](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset)
- **Depth Estimation**: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/hmobinizade/DA-VITON.git
cd DA-VITON
pip install -r requirements.txt
```

## 🗂 Project Structure

```
DA-VITON/
├── configs/             # YAML configs for training/testing
├── datasets/            # Dataset loaders and preprocessing
├── models/              # Network components (encoders, generator, discriminator)
├── utils/               # Losses and helper functions
├── train.py             # Model training
├── test.py              # Inference / evaluation
└── README.md
```

## 🚀 Usage

### 1. Prepare Dataset
- Download the VITON-HD dataset and extract it to the appropriate folder.
- Generate garment depth maps using Depth-Anything-V2.
- Run the garment refinement module to filter irrelevant regions.

### 2. Train the Model

```bash
python train.py --...
```

### 3. Test the Model

```bash
python test.py --...
```

## 📄 Acknowledgements

- The dataset used: [VITON-HD](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset)
- Depth maps generated with: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

---
