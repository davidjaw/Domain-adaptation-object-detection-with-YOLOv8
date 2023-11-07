import os
from utils import visualize, get_img_label_path
import cv2
import torch
import torchvision
import numpy as np


img_paths, label_paths = get_img_label_path('G:/dataset/ACDC/YOLO/train/fog')

for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    labels = []
    with open(label_paths[i], 'r') as f:
        for line in f:
            labels.append(list(map(float, line.split(' '))))

    visualize(img, labels, .7)
