# Dyno-Net: 胃肠道息肉检测的动态特征提取模型

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

[English](README.md) | [中文介绍](README.zh-CN.md)

## 📌 项目简介

本项目为 AISTATS 2026 论文 **"Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection"** 的官方 PyTorch 实现代码。

胃肠道息肉往往是结直肠癌的早期病变。基于实时准确检测的需求，我们提出了 **Dyno-Net**，一个动态特征提取模型框架，它集成了多尺度融合（DynoFPN）、自适应卷积（DynoConv）以及边界优化（RefineDet LSCSBD）。

相比于现有主流检测器，Dyno-Net 实现了：
- **+23.5%** 的息肉目标特征响应强度提升。
- **+17.8%** 的微小及不规则形状息肉检测效果提升。
- **mIoU 平均交并比** 从 0.68 大幅跃升至 0.81。

---

## 🚀 核心贡献
1. **增强型多尺度融合 (DynoFPN):** 动态加权跨尺度特征，从而更好地表示浅层纹理和深层语义的互补特征，将特征融合效率提高 23.5%。
2. **自适应卷积 (DynoConv):** 自适应地将卷积采样与局部轮廓对齐，大幅推动了对于 2~10 毫米、扁平和小叶状息肉的检出率。
3. **自适应边界优化 (RefineDet LSCSBD):** 通过对边界概率分布进行建模，并且补偿统计偏差，显著提高了定位精度。

---

## 🛠️ 安装环境

```bash
# 克隆仓库
git clone https://github.com/ZiYan416/Dyno-Net.git
cd Dyno-Net/Dyno-Net

# 推荐：创建独立虚拟环境
python -m venv venv
source venv/bin/activate  # Windows系统执行: venv\Scripts\activate

# 安装依赖
pip install -e .
```

---

## 📊 数据集准备
项目期望数据集以 YOLO 格式组织（目前在 CVC-ClinicDB, Kvasir-SEG 可直接应用）。请将数据集放置在 `dataset/polyp/` 目录中：

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

## 💻 使用方法

### 训练步骤
使用本仓库中的 `train.py` 开始训练Dyno-Net模型。默认参数将使用 `dataset/polyp.yaml` 加载数据，亦可搭配预训练权重如 `yolo11n.pt` 等初始化模型。

```bash
python train.py --data dataset/polyp/polyp.yaml --epochs 100 --batch-size 16
```
*(如有需要，您也可以在执行命令或配置项中调整超参数。)*

### 评估测试
模型训练完成后，在验证或测试集上评估模型的表现（mAP 以及 mIoU 指标）：
```bash
python val.py --weights runs/train/exp/weights/best.pt --data dataset/polyp/polyp.yaml
```

---

## 📝 引用
如果本仓库对您的研究或开源项目有所帮助，烦请引用我们的AISTATS 2026论文：

```bibtex
@inproceedings{song2026dynonet,
  title={Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection},
  author={Song, Zijie and Wan, Jingjing and Meng, Xianchun and Hua, Qingye and Zhu, Wenjie and Chen, Bolun and Shao, Wei},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  address={Tangier, Morocco}
}
```

## 📄 开源协议
本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 框架构建。根据 YOLO 的开源协议要求，本项目同样采用 **AGPL-3.0 License** 协议发布。详情请查阅该仓库内的 [LICENSE](LICENSE) 文本。

## 🤝 致谢
为了达到优异性能和可拓展性，本框架基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 代码库打造。我们感谢开发者和开源社区为框架做出的贡献。