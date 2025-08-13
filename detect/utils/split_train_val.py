
import os
import random
import shutil
from pathlib import Path

def split_voc_dataset(images_dir, annotations_dir, output_dir, train_ratio=0.8, seed=42):
    """
    划分VOC格式数据集为训练集和验证集
    
    参数:
        images_dir: 原始图片目录路径
        annotations_dir: 原始标注文件目录路径
        output_dir: 输出根目录路径
        train_ratio: 训练集比例(默认0.8)
        seed: 随机种子(默认42)
    """
    # 设置随机种子保证可复现
    random.seed(seed)
    
    # 创建输出目录结构
    output_images = Path(output_dir) / 'images'
    output_labels = Path(output_dir) / 'labels'
    
    (output_images / 'train').mkdir(parents=True, exist_ok=True)
    (output_images / 'val').mkdir(parents=True, exist_ok=True)
    (output_labels / 'train').mkdir(parents=True, exist_ok=True)
    (output_labels / 'val').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件名(不带扩展名)
    image_files = [f.stem for f in Path(images_dir).glob('*.jpg')]
    random.shuffle(image_files)
    
    # 计算划分点
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"总样本数: {len(image_files)}")
    print(f"训练集: {len(train_files)} 验证集: {len(val_files)}")
    
    # 复制文件到对应目录
    for file_stem in train_files:
        # 处理图片文件
        src_img = Path(images_dir) / f"{file_stem}.jpg"
        dst_img = output_images / 'train' / f"{file_stem}.jpg"
        shutil.copy2(src_img, dst_img)
        
        # 处理标注文件
        src_ann = Path(annotations_dir) / f"{file_stem}.txt"
        dst_ann = output_labels / 'train' / f"{file_stem}.txt"
        shutil.copy2(src_ann, dst_ann)
    
    for file_stem in val_files:
        # 处理图片文件
        src_img = Path(images_dir) / f"{file_stem}.jpg"
        dst_img = output_images / 'val' / f"{file_stem}.jpg"
        shutil.copy2(src_img, dst_img)
        
        # 处理标注文件
        src_ann = Path(annotations_dir) / f"{file_stem}.txt"
        dst_ann = output_labels / 'val' / f"{file_stem}.txt"
        shutil.copy2(src_ann, dst_ann)
    
    print("数据集划分完成！")

if __name__ == '__main__':
    # 示例用法
    images_dir = '../datasets/VOC2007/JPEGImages'  # 替换为图片目录
    annotations_dir = '../datasets/VOC2007/Annotations_trans'  # 替换为标注目录
    output_dir = 'VOC2007_split'  # 输出目录
    
    split_voc_dataset(images_dir, annotations_dir, output_dir)
