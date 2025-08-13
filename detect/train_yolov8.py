
from ultralytics import YOLO
import argparse

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='训练设备，如0或0,1,2,3')
    args = parser.parse_args()

    # 加载预训练模型
    model = YOLO(args.weights)
    
    # 训练配置
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'save': True,  # 保存训练检查点
        'save_period': 10,  # 每10个epoch保存一次
        'project': 'runs/train',  # 保存目录
        'name': 'exp',  # 实验名称
        'exist_ok': True,  # 允许覆盖现有实验
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'auto',  # 自动选择优化器
        'lr0': 0.01,  # 初始学习率
        'cos_lr': True,  # 使用余弦学习率调度
    }

    # 开始训练
    results = model.train(**train_args)
    
    # 验证模型
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")  # 打印mAP指标

if __name__ == '__main__':
    main()
