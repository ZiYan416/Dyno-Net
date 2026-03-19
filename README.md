<div align="center">

#  Dyno-Net
### A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-red.svg)]()
[![Conference](https://img.shields.io/badge/Paper-AISTATS%202026-2b9348.svg)]()

[English](README.md) | [¼òÌåÖÐÎÄ](README.zh-CN.md)

</div>

<br>

> **Abstract:** Gastrointestinal polyps are precursors to colorectal cancer, underscoring the need for accurate early detection. We propose **Dyno-Net**, a dynamic feature extraction framework integrating multi-scale fusion (**DynoFPN**), adaptive convolution (**DynoConv**), and boundary refinement (**RefineDet LSCSBD**).

This repository contains the official PyTorch implementation of the AISTATS 2026 paper: *Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection*.

---

##  Key Contributions & Highlights

Gastrointestinal polyp detection is challenging due to complex morphologies, diverse types, and low tissue contrast. **Dyno-Net** addresses these issues through three novel components:

*   ** Enhanced Multi-Scale Fusion (DynoFPN):** Dynamically weights cross-scale features to strengthen the complementary representation of shallow textures and deep semantics.
*   ** Adaptive Convolution (DynoConv):** Adaptively aligns convolutional sampling with local contours, specifically targeting hard-to-detect variations.
*   ** Adaptive Boundary Refinement (RefineDet LSCSBD):** Models boundary probability distributions and compensates for statistical bias to significantly enhance bounding-box precision.

###  Performance Gains over Baseline
| Metric | Baseline | **Dyno-Net (Ours)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean IoU (mIoU)** | 0.68 | **0.81** | `+ 0.13` |
| **Small/Atypical Polyp Detection** | Base | **+17.8%** | `+ 17.8%` |
| **Feature Response Intensity** | Base | **+23.5%** | `+ 23.5%` |

*(Evaluated empirically on benchmark datasets including CVC-ClinicDB and Kvasir-SEG)*

---

##  Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/Dyno-Net.git
cd Dyno-Net
```

**2. Create a virtual environment (Recommended)**
```bash
conda create -n dynonet python=3.10
conda activate dynonet
# Or use venv: python -m venv venv && source venv/bin/activate
```

**3. Install dependencies**
```bash
# Install required packages
pip install -e .
```

---

##  Dataset Preparation

The model expects datasets to be organized in the standard YOLO format. Please place your processed datasets (e.g., CVC-ClinicDB, Kvasir-SEG) in the `dataset/` folder:

```text
dataset/
 polyp/
     polyp.yaml          # Dataset configuration file
     images/
        train/
        val/
        test/
     labels/
         train/
         val/
         test/
```

---

##  Quick Start

###  Training
Train Dyno-Net using the provided `train.py` script. The default configuration relies on `polyp.yaml` for data and uses `yolo11n.pt` or custom weights for initialization.

```bash
python train.py --data dataset/polyp/polyp.yaml --epochs 100 --batch-size 16 --device 0
```

###  Evaluation (Validation / Testing)
Evaluate the trained models on your testing set to get mAP and mIoU metrics:

```bash
python val.py --weights runs/train/exp/weights/best.pt --data dataset/polyp/polyp.yaml --device 0
```

---

##  Citation

If you find this code or our paper useful for your research, please cite our AISTATS 2026 paper:

```bibtex
@inproceedings{song2026dynonet,
  title={Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection},
  author={Song, Zijie and Wan, Jingjing and Meng, Xianchun and Hua, Qingye and Zhu, Wenjie and Chen, Bolun and Shao, Wei},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  address={Tangier, Morocco}
}
```

##  License & Acknowledgments

*   **License:** This project is licensed under the **[GNU AGPL-3.0 License](LICENSE)**. This is in accordance with the license requirements of the Ultralytics YOLO framework upon which this project is built.
*   **Acknowledgments:** Our framework is proudly built on top of the excellent [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) repository. We thank the developers and the open-source community for their foundational work.
