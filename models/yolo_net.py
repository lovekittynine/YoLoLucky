#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:43:16 2022

@author: weishaowei
@description: backbone: vgg16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models


class YoLoNet(nn.Module):
  """
  YoLoNet
  """
  def __init__(self, backbone="vgg16", B=1, C=1, pretrained=""):
    super().__init__()
    # per-grid boxes
    self.B = B
    # number of classes
    self.C = C
    if backbone != "vgg16":
      raise ValueError("only support vgg16 backbone")
    # load vgg16 feature layers
    vgg16 = models.vgg16(pretrained=False)
    # 获取vgg16的卷积层作为特征提取层
    # 特征输出 512x7x7
    self.features = vgg16.features
    # regressor输出层
    # 注意输出归一化到(0-1)
    self.output_channels = B * 5 + C
    self.regressor = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, self.output_channels, 1, 1, 0),
                                   nn.Sigmoid())
    if pretrained:
      print("Loading Parameters from %s"%os.path.abspath(pretrained))
      ckpt = torch.load(pretrained, map_location="cpu")
      # 参数名映射
      ckpt = dict(map(lambda x:(x[0].replace("features.",""), x[1]), ckpt.items()))
      self.features.load_state_dict(ckpt)
      
  def forward(self, xs):
    xs = self.features(xs)
    xs = self.regressor(xs)
    return xs
  
  
if __name__ == "__main__":
  yolo = YoLoNet(pretrained="./vgg16_features.pth")
  print(yolo)
  yolo.eval()
  xs = torch.randn((1, 3, 224, 224))
  ys = yolo(xs)
  print(ys.shape)