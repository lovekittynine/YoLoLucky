#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:31:36 2022

@author: weishaowei
"""

import torch
import cv2
from PIL import Image
import numpy as np
from torch.utils import data
import glob
import os
from xml.etree import ElementTree
from collections import Counter, OrderedDict
from torchvision import transforms
import json


class YoLoDataSet(data.Dataset):
  """
  YoLo车辆数据集:
    return 7x7x(5+C)
  """
  def __init__(self, data_folder="../北京理工车辆数据集", img_size=224, stride=32, batchsize=32, multiscale=False):
    super().__init__()
    # 固定随机种子
    np.random.seed(940806)
    self.data_folder = data_folder
    self.image_folder = os.path.join(self.data_folder, "JPEGImages")
    # self.image_folder = os.path.join(self.data_folder, "Images")
    self.annote_folder = os.path.join(self.data_folder, "Annotations")
    # 记录数据集总类别个数
    self.classes = OrderedDict()
    self.img_size = img_size
    # 网络下采样倍数
    self.stride = stride
    # grid size
    self.grid_size = self.img_size // self.stride
    self.imgpaths = glob.glob(self.image_folder + "/*.jpg")
    self.labelpaths = sorted(glob.glob(self.annote_folder + "/*.xml"), 
                             key=lambda x: int(x.split("/")[-1][:-4]))
    self.__getCategory()
    # 颜色抖动数据增强操作
    self.transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ColorJitter(0.25, 0.4, 0.25, 0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    # 是否启动多尺度训练
    self.multiscale = multiscale
    self.batchsize = batchsize
    self.scales = [224, 256, 288, 320, 352, 384, 416]
    # 计数变量
    self.counter = 0
    
    
    
  def __parse(self, imgpath, labelpath):
    # 解析图像及其bounding boxes list标签
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dom = ElementTree.parse(labelpath)
    # 获取图像分辨率
    H, W, C = img.shape
    bboxes = []
    for obj in dom.findall("object"):
      category = obj.find("name").text
      cls_id = self.classes[category]
      # get bnd box
      bndbox = obj.find("bndbox")
      xmin = int(bndbox.find("xmin").text)
      ymin = int(bndbox.find("ymin").text)
      xmax = int(bndbox.find("xmax").text)
      ymax = int(bndbox.find("ymax").text)
      bboxes.append([xmin, ymin, xmax, ymax, cls_id])
    return img, (H, W), bboxes
        
  
  def __getCategory(self):
    # 获取数据集类别数
    categorys = []
    for file in self.labelpaths:
      dom = ElementTree.parse(file)
      for obj in dom.findall("object"):
        category = obj.find("name").text
        categorys.append(category)
        if category not in self.classes:
          self.classes[category] = len(self.classes)
    counter = Counter(categorys)
    print("数据集类别分布:", counter)
    print(self.classes)
    with open("../voc_eval/voc_classes.json", "w") as f:
      json.dump(self.classes, f)
    
    
  def __getitem__(self, idx):
    imgpath = self.imgpaths[idx]
    labelpath = os.path.join(self.annote_folder, os.path.basename(imgpath).replace("jpg", "xml"))
    image, (H, W), bboxes = self.__parse(imgpath, labelpath)
    if np.random.rand() >= 0.5:
      # 随机裁剪增强
      image, (H, W), bboxes = self.random_crop_img_bboxes(image, bboxes)
    # 每10个batch随机切换训练尺度
    if self.multiscale and self.counter == self.batchsize*10:
      self.counter = 0
      # 随机选择新的尺度
      np.random.shuffle(self.scales)
      self.img_size = self.scales[self.counter%len(self.scales)]
      self.grid_size = self.img_size // self.stride
    # resize image
    image = cv2.resize(image, (self.img_size, self.img_size))
    h_ratio, w_ratio = self.img_size/H, self.img_size/W
    # ground truth label
    label = np.zeros((self.grid_size, self.grid_size, 5+len(self.classes)), dtype=np.float32)
    mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
    
    for box in bboxes:
      xmin, ymin, xmax, ymax, cls_id = box
      # box resize
      xmin *= w_ratio
      xmax *= w_ratio
      ymin *= h_ratio
      ymax *= h_ratio
      # convert to (x,y,w,h)
      width = xmax - xmin
      height = ymax - ymin
      # 坐标32倍下采样
      c_x = (xmin + 0.5 * width) / 32
      c_y = (ymin + 0.5 * height) / 32
      # 随机左右反转进行数据增强
      if np.random.rand() >= 0.5:
        c_x = self.grid_size - c_x
        image = image[:, ::-1, :].copy() 
      # print(c_x, c_y)
      # 计算中心点网格坐标
      grid_x, grid_y = int(np.floor(c_x)), int(np.floor(c_y))
      # print(grid_x, grid_y)
      offset_x, offset_y = c_x - grid_x, c_y - grid_y
      # normalize width and height
      width /= self.img_size
      height /= self.img_size
      # set label
      label[grid_y, grid_x, 0] = offset_x
      label[grid_y, grid_x, 1] = offset_y
      label[grid_y, grid_x, 2] = width
      label[grid_y, grid_x, 3] = height
      label[grid_y, grid_x, 4] = 1.0
      label[grid_y, grid_x, 5+cls_id] = 1.0
      # set mask
      mask[grid_y, grid_x] = 1.0  
    
    # convert to tensor
    image = self.transform(image)
    label = torch.from_numpy(label)
    mask = torch.from_numpy(mask)
    # update计数变量
    self.counter += 1
    
    return image, label, mask, self.img_size
  
  
  def __len__(self):
    return len(self.imgpaths)
  
  
  def display_bbox(self, image, label):
    """
    note:该函数作用于transform之前
    """
    (row_grids, col_grids) = np.where(label[..., 4]==1.0)
    # restore bbox
    for y, x in zip(row_grids, col_grids):
      c_x, c_y = x + label[y, x, 0], y + label[y, x, 1]
      width = label[y, x, 2] * self.img_size
      height = label[y, x, 3] * self.img_size
      c_x *= 32
      c_y *= 32
      xmin, xmax = int(c_x - 0.5 * width), int(c_x + 0.5 * width)
      ymin, ymax = int(c_y - 0.5 * height), int(c_y + 0.5 * height)
      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, 2)
    # cv2.imshow("img", image)
    # cv2.destroyAllWindows()
    cv2.imwrite("./test.jpg", image)
    
    
  def display_bboxv2(self, image, label):
    y, x = torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size))
    # 7x7x4
    bboxes = label[..., :4]
    # restore center coordinate
    c_x = (bboxes[..., 0] + x) * self.stride
    c_y = (bboxes[..., 1] + y) * self.stride
    width = bboxes[..., 2] * self.img_size
    height = bboxes[..., 3] * self.img_size
    # 7x7
    xmin = c_x - 0.5 * width
    xmax = c_x + 0.5 * width
    ymin = c_y - 0.5 * height
    ymax = c_y + 0.5 * height
    # find objects
    (row_grids, col_grids) = torch.where(label[..., 4]==1.0)
    mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    image = (image * std + mean) * 255.0
    image = image.permute([1,2,0]).contiguous()
    image = image.numpy().astype(np.uint8)
    for grid_y, grid_x in zip(row_grids, col_grids):
      x1, y1, x2, y2 = xmin[grid_y, grid_x].int().item(), ymin[grid_y, grid_x].int().item(), xmax[grid_y, grid_x].int().item(), ymax[grid_y, grid_x].int().item()
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
    cv2.imwrite("./test.jpg", image)  
    
    
  def random_crop_img_bboxes(self, image, bboxes):
    # 随机裁剪图像和bbox
    height, width = image.shape[:2]
    # 包含所有目标的最小bbox坐标初始化
    xmin = width
    xmax = 0
    ymin = height
    ymax = 0
    for bbox in bboxes:
      xmin = min(bbox[0], xmin)
      ymin = min(bbox[1], ymin)
      xmax = max(bbox[2], xmax)
      ymax = max(bbox[3], ymax)
      
    # 裁剪的框到边界的距离[最大值]
    crop_left = xmin
    crop_right = width - xmax
    crop_top = ymin
    crop_bottom = height - ymax
    # 随机调整裁剪框
    crop_xmin = int(xmin - np.random.uniform(0, crop_left))
    crop_ymin = int(ymin - np.random.uniform(0, crop_top))
    crop_xmax = int(xmax + np.random.uniform(0, crop_right))
    crop_ymax = int(ymax + np.random.uniform(0, crop_bottom))
    
    # 边界调整
    crop_xmin = max(0, crop_xmin)
    crop_ymin = max(0, crop_ymin)
    crop_xmax = min(width, crop_xmax)
    crop_ymax = min(height, crop_ymax)
    # crop image and  bboxes
    image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    bboxes_crop = []
    for bbox in bboxes:
      bboxes_crop.append([bbox[0]-crop_xmin, bbox[1]-crop_ymin, bbox[2]-crop_xmin, bbox[3]-crop_ymin, bbox[4]])
    H, W = image.shape[:2]
    return image, (H, W), bboxes_crop
    
      
    
    
    
if __name__ == "__main__":
  yolodataset = YoLoDataSet(img_size=448, multiscale=True)
  print(yolodataset.classes)
  for image, label, mask, imgsize in iter(yolodataset):
    print(image.shape)
    break
    pass
  # print(image.shape)
  # print(label)
  print(mask)
  yolodataset.display_bboxv2(image, label)
    
    
    
    
    





