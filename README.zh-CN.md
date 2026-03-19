<div align="center">

#  Dyno-Net
### θ����Ϣ����Ķ�̬������ȡģ��

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-red.svg)]()
[![Conference](https://img.shields.io/badge/Paper-AISTATS%202026-2b9348.svg)]()

[English](README.md) | [��������](README.zh-CN.md)

</div>

<br>

> **ժҪ��** θ����Ϣ�������ǽ�ֱ���������ڲ��䡣Ϊ�������ٴ��϶���ʵʱ���߾��ȼ�������������������� **Dyno-Net**һ���ں��˶�߶ȶ�̬�ںϣ�**DynoFPN**��������Ӧ������**DynoConv**���ͱ߽��Ż���**RefineDet LSCSBD**���Ķ�̬������ȡģ�Ϳ�ܡ�

����ĿΪ AISTATS 2026 ���� **"Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection"** �Ĺٷ� PyTorch ʵ�ִ��롣

---

##  ���Ĵ���������

�����������ģ���ڴ���������̬Ϣ�⼰��֯�Աȶȵ͵������ϵľ����ԣ�**Dyno-Net** �������������ģ�飬��������˼���³�����붨λ���ȣ�

*   ** ��ǿ�Ͷ�߶��ں� (DynoFPN):** ��̬��Ȩ��߶����������õر���ǳ���������������Ļ������ԡ�
*   ** ����Ӧ���� (DynoConv):** ����Ӧ�ؽ�����������ֲ��������룬������СϢ�⡢��ƽ�Լ�СҶ״Ϣ�����Ա�������������⡣
*   ** ����Ӧ�߽��Ż� (RefineDet LSCSBD):** ͨ���Ա߽���ʷֲ����н�ģ������ͳ��ƫ���������˱߽��Ķ�λ���ȡ�

###  ģ�����ܶԱ�����
| ����ָ�� | ����ģ�� (Baseline) | **Dyno-Net (����Ŀ)** | ���� |
| :--- | :---: | :---: | :---: |
| **ƽ�������� (mIoU)** | 0.68 | **0.81** | `+ 0.13` |
| **΢С/������Ϣ������** | ��׼ | **+17.8%** | `+ 17.8%` |
| **Ŀ��������Ӧǿ��** | ��׼ | **+23.5%** | `+ 23.5%` |

*(ע�����ݻ��� CVC-ClinicDB �� Kvasir-SEG ���������ݼ�����ʵ����֤�ó���)*

---

##  ������װ

**1. ��¡�ֿ�**
```bash
git clone https://github.com/ZiYan416/Dyno-Net.git
cd Dyno-Net
```

**2. �������⻷�� (�Ƽ�)**
```bash
conda create -n dynonet python=3.10
conda activate dynonet
# ��ʹ�� venv ����: python -m venv venv && source venv/bin/activate
```

**3. ��װ����**
```bash
pip install -e .
```

---

##  ���ݼ�׼��

��Ŀ�������ݼ�������� YOLO ��ʽ��֯����ȷ���Ѵ����õ����ݼ�λ����Ӧ�� `dataset/` Ŀ¼�У�

```text
dataset/
 polyp/
     polyp.yaml          # ���ݼ������ļ�
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

##  ���ٿ�ʼ

###  ģ��ѵ��
ʹ�òֿ��ṩ�� `train.py` ��ʼѵ��Dyno-Netģ�͡�Ĭ�ϲ�����ʹ�� `polyp.yaml` �������ݣ���ɴ���Ԥѵ��Ȩ���� `yolo11n.pt` ʹ�ã�

```bash
python train.py --data dataset/polyp/polyp.yaml --epochs 100 --batch-size 16 --device 0
```

###  ��֤������
ģ��ѵ����ɺ�����������֤������Լ�����������ȡ mAP �Լ� mIoU ָ�꣺

```bash
python val.py --weights runs/train/exp/weights/best.pt --data dataset/polyp/polyp.yaml --device 0
```

---

##  ��������

������ֿ�Ĵ���������ǵ�����˼��������о���Դ��Ŀ���������������������ǵ� AISTATS 2026 ���ģ�

```bibtex
@inproceedings{song2026dynonet,
  title={Dyno-Net: A Dynamic Feature Extraction Model for Gastrointestinal Polyp Detection},
  author={Song, Zijie and Wan, Jingjing and Meng, Xianchun and Hua, Qingye and Zhu, Wenjie and Chen, Bolun and Shao, Wei},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  address={Tangier, Morocco}
}
```

##  ��ԴЭ������л

*   **��ԴЭ��:** ����Ŀ���� [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) ��ܹ��������� YOLO ��Ҫ����ؿ�Դ���񣬱���Ŀͬ������ **[GNU AGPL-3.0 License](LICENSE)** Э�鷢����
*   **��л:** Ϊ�˴ﵽ�������ܺͿ���չ�ԣ�����ܵײ��������� [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) �������죬���������еĿ����ߺͿ�Դ�����¾���
