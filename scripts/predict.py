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
import copy
from tqdm import tqdm


parser = argparse.ArgumentParser("YoLoLucky Predict")
parser.add_argument("--ckpt", default="../checkpoint/epoch_71.pt", type=str)
parser.add_argument("--image", default="", type=str)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--boxes", default=1, type=int)
parser.add_argument("--num_classes", default=7, type=int)
parser.add_argument("--backbone", default="vgg16", type=str)
parser.add_argument("--nms_thresh", default=0.5, type=float)
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
    # self.idx2cls = {0:"truck",1:"car",2:"bus",3:"microbus",4:"minivan",5:"suv",6:"microvan"}
    # voc类别索引
    self.idx2cls = {0:"chair",1:"car",2:"horse",3:"person",4:"bicycle",5:"cat",6:"dog",
                    7:"train",8:"aeroplane",9:"diningtable",10:"tvmonitor",11:"bird",12:"bottle",
                    13:"motorbike",14:"pottedplant",15:"boat",16:"sofa",17:"sheep",18:"cow",19:"bus"}
    
    
    
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
  def predict(self, visualize=True):
    img, (h_ratio, w_ratio), raw_img = self.preprocess()
    H, W, _ = raw_img.shape
    preds = self.model(img).squeeze().detach().cpu()
    bboxes = []
    # 中心点预测
    grid_y, grid_x = torch.where(preds[4,:,:] >= 0.3)
    
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
        if p >= 0.5:
          # covert xywh to xyxy
          x1 = max(0, int(x - 0.5*w))
          y1 = max(0, int(y - 0.5*h))
          x2 = min(W, int(x + 0.5*w))
          y2 = min(H, int(y + 0.5*h))
          box = [x1, y1, x2, y2, p, self.idx2cls[idx]]
          bboxes.append(box)
    
    # display
    # nms过滤
    if len(bboxes) > 0:
      bboxes = self.nms(bboxes)
    if visualize:
      self.display_detects(raw_img, bboxes)
    return bboxes
     
     
     
  def display_detects(self, image, bboxes, filename="../detected.jpg"):
    height, width, _ = image.shape
    for box in bboxes:
      if box:
        x1, y1, x2, y2, p, category = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        del_w, del_h = min(50, int(width)), min(30, int(height))
        # print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x1+del_w, y1+del_h), (0, 255, 0), -1, 1)
        cv2.putText(image, "%s:%.3f"%(category,p), (x1, y1+15), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0,0,255), 1)
    plt.imshow(image)
    plt.show()
    cv2.imwrite(filename, image[..., ::-1])
    
    
  def nms(self, bboxes):
    # nms后处理[可以进一步升级为soft-nms,大于nms_thresh的box得分score进行降权而不是直接丢弃]
    bbox_category = {}
    # 对每一类的bbox进行统计
    for box in bboxes:
      cname = box[-1]
      if cname not in bbox_category:
        bbox_category[cname] = [box]
      else:
        bbox_category[cname].append(box)
    # print(bbox_category)
    bboxes_keep = []
    # 分别对每一类的bbox进行nms
    for c_bboxes in bbox_category.values():
      # 按照类别概率进行排序[由大到小]
      bboxes_candidate = sorted(c_bboxes, key=lambda x:x[-2], reverse=True)
      while len(bboxes_candidate) > 0:
        anchor_box = bboxes_candidate[0]
        bboxes_keep.append(anchor_box)
        bboxes_candidate = bboxes_candidate[1:]
        bboxes_candidate_keep = []
        for bbox in bboxes_candidate:
          # compute iou
          iou = self.calculate_iou(bbox[:4], anchor_box[:4])
          # print(iou)
          if iou <= args.nms_thresh:
            bboxes_candidate_keep.append(bbox)
        bboxes_candidate = copy.deepcopy(bboxes_candidate_keep)
        
    return bboxes_keep
      
  
  def calculate_iou(self, box1, box2):
    box1_area = max((box1[2]-box1[0])*(box1[3]-box1[1]), 0.0)
    box2_area = max((box2[2]-box2[0])*(box2[3]-box2[1]), 0.0)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0.0, x2-x1) * max(0.0, y2-y1)
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-8)
    return iou
      
      
    

if __name__ == "__main__":
  args.ckpt = "../checkpoint/epoch_multiscale.pt"
  args.num_classes = 20
  detector = YoLoLuckyPredictor()
  args.img_size = 224
  args.image = "../VOCdevkit_test/VOC2007/JPEGImages/000018.jpg"
  # args.image = "../utils/test1.jpeg"
  bbox = detector.predict()
  print(bbox)
  
  """
  # voc 2007测试集评测
  testFolder = "../VOCdevkit_test/VOC2007/JPEGImages"
  cls_preds = {}
  for file in tqdm(os.listdir(testFolder), desc="VOC 2007 Testing..."):
    image_id = file[:-4]
    imgpath = os.path.join(testFolder, file)
    args.image = imgpath
    bboxes = detector.predict(visualize=False)
    for box in bboxes:
      bnd = box[:4]
      prob = box[4]
      cname = box[-1]
      if cname not in cls_preds:
        cls_preds[cname] = [[image_id, prob] + bnd]
      else:
        cls_preds[cname].append([image_id, prob] + bnd)
        
  # print(cls_preds)     
  # save detection results
  detectionFolder = "../voc_eval/detections"
  for cname, bboxes in cls_preds.items():
    with open(os.path.join(detectionFolder, "%s.txt"%cname), "w") as f:
      for box in bboxes:
        f.write("%s %.3f %f %f %f %f\n"%(box[0], box[1], box[2], box[3], box[4], box[5]))
  """
  
    

