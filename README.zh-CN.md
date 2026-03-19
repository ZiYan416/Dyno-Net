<div align="center">

#  Dyno-Net
### 胃肠道息肉检测的动态特征提取模型

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-red.svg)]()
[![Conference](https://img.shields.io/badge/Paper-AISTATS%202026-2b9348.svg)]()

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

<br>

> **摘要：** 胃肠道息肉往往是结直肠癌的早期病变。为了满足临床上对于实时、高精度检测的迫切需求，我们提出了 **Dyno-Net**一个融合了多尺度动态融合（**DynoFPN**）、自适应卷积（**DynoConv**）和边界优化（**RefineDet LSCSBD**）的动态特征提取模型框架。

本项目为 AISTATS 2026 论文 **"Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection"** 的官方 PyTorch 实现代码。

---

##  核心创新与优势

针对现有主流模型在处理复杂形态息肉及组织对比度低等问题上的局限性，**Dyno-Net** 引入了三大核心模块，大幅提升了检测的鲁棒性与定位精度：

*   ** 增强型多尺度融合 (DynoFPN):** 动态加权跨尺度特征，更好地表征浅层纹理和深层语义的互补特性。
*   ** 自适应卷积 (DynoConv):** 自适应地将卷积采样与局部轮廓对齐，攻克了小息肉、扁平以及小叶状息肉难以被特征捕获的问题。
*   ** 自适应边界优化 (RefineDet LSCSBD):** 通过对边界概率分布进行建模并补偿统计偏差，显著提高了边界框的定位精度。

###  模型性能对比提升
| 核心指标 | 基线模型 (Baseline) | **Dyno-Net (本项目)** | 增益 |
| :--- | :---: | :---: | :---: |
| **平均交并比 (mIoU)** | 0.68 | **0.81** | `+ 0.13` |
| **微小/不规则息肉检测率** | 基准 | **+17.8%** | `+ 17.8%` |
| **目标特征响应强度** | 基准 | **+23.5%** | `+ 23.5%` |

*(注：数据基于 CVC-ClinicDB 与 Kvasir-SEG 等主流数据集多组实验验证得出。)*

---

##  环境安装

**1. 克隆仓库**
```bash
git clone https://github.com/yourusername/Dyno-Net.git
cd Dyno-Net
```

**2. 创建虚拟环境 (推荐)**
```bash
conda create -n dynonet python=3.10
conda activate dynonet
# 或使用 venv 环境: python -m venv venv && source venv/bin/activate
```

**3. 安装依赖**
```bash
pip install -e .
```

---

##  数据集准备

项目期望数据集以最常见的 YOLO 格式组织，请确保已处理好的数据集位于相应的 `dataset/` 目录中：

```text
dataset/
 polyp/
     polyp.yaml          # 数据集配置文件
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

##  快速开始

###  模型训练
使用仓库提供的 `train.py` 开始训练Dyno-Net模型。默认参数将使用 `polyp.yaml` 加载数据，亦可搭配预训练权重如 `yolo11n.pt` 使用：

```bash
python train.py --data dataset/polyp/polyp.yaml --epochs 100 --batch-size 16 --device 0
```

###  验证与评估
模型训练完成后，您可以在验证集或测试集上评估并获取 mAP 以及 mIoU 指标：

```bash
python val.py --weights runs/train/exp/weights/best.pt --data dataset/polyp/polyp.yaml --device 0
```

---

##  引用声明

如果本仓库的代码或者我们的论文思想对您的研究或开源项目有所帮助，烦请引用我们的 AISTATS 2026 论文：

```bibtex
@inproceedings{song2026dynonet,
  title={Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection},
  author={Song, Zijie and Wan, Jingjing and Meng, Xianchun and Hua, Qingye and Zhu, Wenjie and Chen, Bolun and Shao, Wei},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  address={Tangier, Morocco}
}
```

##  开源协议与致谢

*   **开源协议:** 本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 框架构建。根据 YOLO 的要求及相关开源精神，本项目同样采用 **[GNU AGPL-3.0 License](LICENSE)** 协议发布。
*   **致谢:** 为了达到优异性能和可拓展性，本框架底层基于优秀的 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 代码库打造，我们向所有的开发者和开源社区致敬。
