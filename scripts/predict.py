#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:51:52 2022

@author: weishaowei
"""

import torch
import os
import sys
sys.path.append("../")
from models.yolo_net import YoLoNet
from skimage import io
from skimage import transform
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser("YoLoLucky Predict")
parser.add_argument("--ckpt", default="../checkpoint/epoch_71.pt", type=str)
parser.add_argument("--image", default="", type=str)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--boxes", default=1, type=int)
parser.add_argument("--num_classes", default=7, type=int)
parser.add_argument("--backbone", default="vgg16", type=str)
args = parser.parse_args()


class YoLoLuckyPredictor():
  """
  预测主类
  """
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build model
    self.model = YoLoNet(args.backbone, args.boxes, args.num_classes)
    # restore parameter
    if args.ckpt:
      self.model.load_state_dict(torch.load(args.ckpt, self.device))
      print("Loading Parameters from %s"%args.ckpt)
    self.model.to(self.device)
    self.model.eval()
    self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    self.idx2cls = {0:"truck",1:"car",2:"bus",3:"microbus",4:"minivan",5:"suv",6:"microvan"}
    
    
    
  def preprocess(self):
    # preprocess image
    raw_img = io.imread(args.image)
    h, w, _ = raw_img.shape
    h_ratio, w_ratio = args.img_size / h, args.img_size / w
    # resize and normalize to 0-1
    img = transform.resize(raw_img, (args.img_size, args.img_size))
    img = torch.from_numpy(img)
    img = img.permute([2,0,1])
    # torch normalize
    img = (img - self.mean)/(self.std + 1e-8)
    # 1x3x224x224
    img = img.unsqueeze(0).to(self.device).float()
    return img, (h_ratio, w_ratio), raw_img
  
  
  @torch.no_grad()
  def predict(self):
    img, (h_ratio, w_ratio), raw_img = self.preprocess()
    preds = self.model(img).squeeze().detach().cpu()
    bboxes = []
    # 中心点预测
    grid_y, grid_x = torch.where(preds[4,:,:] >= 0.5)
    
    if len(grid_y) > 0:
      offset_x, offset_y = preds[0,grid_y,grid_x], preds[1,grid_y,grid_x]
      # 得到尺度
      width = preds[2,grid_y,grid_x] * args.img_size / w_ratio
      height = preds[3,grid_y,grid_x] * args.img_size / h_ratio
      # 得到前景概率
      confidence = preds[4,grid_y,grid_x]
      # 得到目标类别概率
      cls_prob = confidence * preds[5:,grid_y,grid_x]
      # 目标类别和索引
      # print(cls_prob.t())
      tgt_prob, tgt_ind = torch.max(cls_prob, dim=0)
      c_x = (offset_x + grid_x) * 32 / w_ratio
      c_y = (offset_y + grid_y) * 32 / h_ratio
      for x, y, w, h, p, idx in zip(c_x.numpy(), c_y.numpy(), width.numpy(), height.numpy(), tgt_prob.numpy(), tgt_ind.numpy()):
        box = [x, y, w, h, p, self.idx2cls[idx]]
        bboxes.append(box)
    else:
      bboxes.append([])
    # display
    # print(bboxes)
    self.display_detects(raw_img, bboxes)
     
     
     
  def display_detects(self, image, bboxes, filename="../detected.jpg"):
    height, width, _ = image.shape
    for box in bboxes:
      if box:
        x, y, w, h, p, category = box
        x1, y1 = max(0, int(x - w//2)), max(0, int(y - h//2))
        x2, y2 = min(width, int(x + w//2)), min(height, int(y + h//2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        del_w, del_h = min(200, int(width)), min(100, int(height))
        # print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x1+del_w, y1+del_h), (0, 255, 0), -1, 1)
        cv2.putText(image, "%s:%.3f"%(category,p), (x1, y1+50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)
    plt.imshow(image)
    plt.show()
    cv2.imwrite(filename, image)
      
      
      
    

if __name__ == "__main__":
  args.image = "../北京理工车辆数据集/Images/002100.jpg"
  args.ckpt = "../checkpoint/epoch_100.pt"
  detector = YoLoLuckyPredictor()
  detector.predict()
  
    
    
    
    

