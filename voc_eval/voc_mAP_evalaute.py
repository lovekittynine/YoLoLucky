#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:12:14 2022

@author: weishaowei
"""

# 实现mAP评测

import xml.etree.ElementTree as ET
import os
import numpy as np
import json
from pprint import pprint


class PASCALEvaluate():
  """
  pascal评测
  """
  def __init__(self, dataFolder="../VOCdevkit_test/VOC2007", detectionFolder="./detections", iou_thresh=0.5):
    self.dataFolder = dataFolder
    self.annotateFolder = os.path.join(self.dataFolder, "Annotations")
    self.detectionFolder = detectionFolder
    self.iou_thresh = iou_thresh
    self.labelPath = os.path.join(self.detectionFolder, "gt.json")
    self.classes = ["bottle", "person", "boat", "chair", "cat", "car", "horse",
                    "aeroplane", "diningtable", "cow", "train", "sofa", "pottedplant",
                    "bird", "bicycle", "tvmonitor", "dog", "motorbike", "bus", "sheep"]
    pass
  
  
  
  def parse_voc(self, labelPath):
    # 解析label中的信息
    xml = ET.parse(labelPath)
    objects = []
    for obj in xml.findall("object"):
      struct_obj = {}
      cname = obj.find("name").text
      struct_obj["name"] = cname
      # find bounding box
      bndbox = obj.find("bndbox")
      xmin = int(bndbox.find("xmin").text)
      xmax = int(bndbox.find("xmax").text)
      ymin = int(bndbox.find("ymin").text)
      ymax = int(bndbox.find("ymax").text)
      struct_obj["bndbox"] = [xmin, ymin, xmax, ymax]
      objects.append(struct_obj)
    return objects
  
  
  
  def calculate_AP(self, precision, recall):
    # 计算每一类的AP值
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    # 对于precision进行平滑
    for i in range(precision.size-1, 0, -1):
      precision[i-1] = np.maximum(precision[i-1], precision[i])
    # 找到recall增加的point索引
    inds = np.where(recall[1:]!=recall[:-1])[0]
    ap = np.sum((recall[inds+1] - recall[inds]) * precision[inds+1])
    return ap
    
  
  
  def voc_cls_AP(self, classname):
    """
    计算给定的voc类别ap:
      @param: classname[voc类名]
      @labelPath: ground truth边界框文件路径
    """
    # 当前类下正样本检测实例数
    npos = 0
    # 获取给定类别下ground truth边界框
    cls_gt_bboxes = {}
    all_objects = []
    for path in os.listdir(self.annotateFolder):
      image_id = path[:-4]
      objects = self.parse_voc(os.path.join(self.annotateFolder, path))
      all_objects.append(objects)
      bboxes = []
      for obj in objects:
        if obj["name"] == classname:
          npos += 1
          bboxes.append(obj["bndbox"])
      # 每个ground truth是否已经被预测的标志位
      dets = [False] * len(bboxes)
      if len(bboxes) > 0:
        cls_gt_bboxes[image_id] = {}
        cls_gt_bboxes[image_id]["bboxes"] = bboxes
        cls_gt_bboxes[image_id]["dets"] = dets
    
    # 获取给定类别下检测结果
    detectionPath = os.path.join(self.detectionFolder, classname+".txt")
    pred_bboxes = []
    with open(detectionPath, "r") as f:
      for line in f:
        line = line.strip().split(" ")
        image_id = line[0]
        score = float(line[1])
        bbox = list(map(lambda x:int(float(x)), line[2:]))
        pred_bboxes.append([image_id, score] + bbox)
    
    fp = [0] * len(pred_bboxes)
    tp = [0] * len(pred_bboxes)
    # pred bbox按照得分降序排列
    pred_bboxes = sorted(pred_bboxes, key=lambda x:x[1], reverse=True)
    # print(pred_bboxes[:5])
    for i, bbox in enumerate(pred_bboxes):
      bndbox = bbox[2:]
      # print(bndbox)
      image_id = bbox[0]
      if image_id not in cls_gt_bboxes:
        fp[i] = 1
        continue
      gt_bboxes = np.array(cls_gt_bboxes[image_id]["bboxes"])
      # print(bndbox, gt_bboxes)
      
      # print(gt_bboxes.shape)
      # compute iou
      x1 = np.maximum(gt_bboxes[:, 0], bndbox[0])
      y1 = np.maximum(gt_bboxes[:, 1], bndbox[1])
      x2 = np.minimum(gt_bboxes[:, 2], bndbox[2])
      y2 = np.minimum(gt_bboxes[:, 3], bndbox[3])
      inter = np.maximum(x2-x1+1, 0) * np.maximum(y2-y1+1, 0)
      union = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1) +\
        (bndbox[2]-bndbox[0]+1) * (bndbox[3]-bndbox[1]+1) - inter
      iou = inter/(union+1e-8)
      # 寻找最大iou
      idx = np.argmax(iou)
      max_iou = iou[idx]
      if max_iou >= self.iou_thresh:
        if not cls_gt_bboxes[image_id]["dets"][idx]:
          tp[i] = 1
          cls_gt_bboxes[image_id]["dets"][idx] = True
        else:
          fp[i] = 1
      else:
        fp[i] = 1
        
    # 计算每个样本处的precision以及recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / npos
    precision = tp / (tp + fp + 1e-8)
    # 计算当前类别下ap
    ap = self.calculate_AP(precision, recall)
    print("finish %s"%classname)
    return ap
  
  
  def get_voc_mAP(self):
    mAP = 0
    aps = {}
    for cname in self.classes:
      ap = self.voc_cls_AP(cname)
      aps[cname] = ap
      mAP += ap
    mAP /= len(self.classes)
    pprint(aps)
    pprint("pascal voc mAP: %.3f"%mAP)
    
        
    



if __name__ == "__main__":
  evalaute = PASCALEvaluate()
  evalaute.get_voc_mAP()
  
  




