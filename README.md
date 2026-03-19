# Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-red)]()

[English](README.md) | [中文介绍](README.zh-CN.md)

## 📌 Introduction
This is the official PyTorch implementation of the AISTATS 2026 paper: **Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection**.

Gastrointestinal polyps are precursors to colorectal cancer, underscoring the need for accurate early detection. We propose **Dyno-Net**, a dynamic feature extraction framework integrating multi-scale fusion (DynoFPN), adaptive convolution (DynoConv), and boundary refinement (RefineDet LSCSBD).

Compared to baseline models, Dyno-Net achieves:
- **+23.5%** higher feature response intensity on polyp targets.
- **+17.8%** better detection of small/atypical polyps.
- **mIoU improvement** from 0.68 to 0.81.

---

## 🚀 Key Contributions
1. **Enhanced Multi-Scale Fusion (DynoFPN):** Dynamically weights cross-scale features to strengthen complementary representation of shallow textures and deep semantics.
2. **Adaptive Convolution (DynoConv):** Adaptively aligns convolutional sampling with local contours, boosting the detection of 2–10 mm, flat, and lobulated polyps.
3. **Adaptive Boundary Refinement (RefineDet LSCSBD):** Models boundary probability distributions and compensates for statistical bias to significantly enhance localization accuracy.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ZiYan416/Dyno-Net.git
cd Dyno-Net/Dyno-Net

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r pyproject.toml # or if using requirements.txt
pip install -e .
```

---

## 📊 Dataset Preparation
The project expects datasets (e.g., CVC-ClinicDB, Kvasir-SEG) to be organized in YOLO format. Please place your datasets in the `dataset/` folder as follows:

```
dataset/
    polyp/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        polyp.yaml
```

---

## 💻 Usage

### Training
You can train Dyno-Net using the provided `train.py` script. The default configuration uses `polyp.yaml` for data and `yolo11n.pt` / custom weights for initialization.

```bash
python train.py --data dataset/polyp/polyp.yaml --epochs 100 --batch-size 16
```
*(Optionally modify hyperparameters in `train.py` or configuration files.)*

### Inference / Evaluation
Evaluate the trained models on your testing set to get mAP and mIoU metrics:
```bash
# Run validation loop
python val.py --weights runs/train/exp/weights/best.pt --data dataset/polyp/polyp.yaml
```

---

## 📝 Citation
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

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🤝 Acknowledgments
Our code is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. We thank the developers and the open-source community for their excellent work.