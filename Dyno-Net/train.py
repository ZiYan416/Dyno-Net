import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics\cfg\models\11\yolo11-Dyno-Net.yaml')
    model.train(data=r'polyp.yaml',  # 数据集文件的位置
                imgsz=640,  # 输入图像的尺寸
                epochs=200,  # 训练的轮数
                batch=16,  # 每个批次中的图像数量
                workers=0,
                device='0',
                optimizer='SGD',  # 优化器
                close_mosaic=10,  #  禁用mosaic增强的最后第10个轮次
                single_cls=False,  # 是否单类别检测
                cache=False,
                cos_lr=True,  # 使用余弦学习率调度器
                lr0=0.01,  # 单独设置初始学习率
                amp=True,  #  是否使用自动混合精度（Automatic Mixed Precision，AMP）训练
                project='runs/train',
                name='exp',
                )

   # # 断点续训启动方式,这个绝对路径就是上文中提到的last.pt文件路径
   # model = YOLO(r"A:\yolov11\yolov11-RCSOSA\runs\train\exp\weights\last.pt")
   # # 中断训练的权重文件中的last.pt
   # results = model.train(resume=True)