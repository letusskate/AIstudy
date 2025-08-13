import xml.etree.ElementTree as ET
import os
from pathlib import Path

# VOC类别
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_file, output_dir):
    in_file = open(xml_file)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    out_file = open(os.path.join(output_dir, Path(xml_file).stem + '.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# 使用示例
annotations_dir = '../datasets/VOC2007/Annotations'
labels_dir = '../datasets/VOC2007/Annotations_trans'
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        convert_annotation(os.path.join(annotations_dir, xml_file), labels_dir)
