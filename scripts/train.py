#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:52:29 2022

@author: weishaowei
"""

import torch
from torch import optim
import os
import sys
sys.path.append("../")

from models.yolo_net import YoLoNet
from dataloader.YoLoDataLoader import YoLoDataSet
from utils.show_detects import display_detects
from torch.utils import data
import argparse
import numpy as np



parser = argparse.ArgumentParser("LuckyYoLo Training")
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--epochs", default=160, type=int)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--warmup_epochs", default=20, type=int)
parser.add_argument("--wd", default=5e-3, type=int)
parser.add_argument("--checkpoint", default="../checkpoint", type=str)
parser.add_argument("--logFolder", default="../log", type=str)
parser.add_argument("--boxes", default=1, type=int)
parser.add_argument("--num_classes", default=7, type=int)
parser.add_argument("--backbone", default="vgg16", type=str)
parser.add_argument("--backbone_pretrained", default="../models/vgg16_features.pth", type=str)
parser.add_argument("--trainFolder", default="../北京理工车辆数据集", type=str)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--multiscale", default="False", type=str, help="是否开启多尺度训练")
args = parser.parse_args()


torch.autograd.set_detect_anomaly(True)

class LuckyYoLoTrainer():
  """
  训练主类
  """
  def __init__(self):
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.logFolder, exist_ok=True)
    self.imgLogFolder = os.path.join(args.logFolder, "images")
    os.makedirs(self.imgLogFolder, exist_ok=True)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build 模型
    self.build_model()
    # create dataloader
    self.create_dataloader()
    # create优化器
    self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
    self.global_step = 0
    self.__cls2id = self.dataset.classes
    self.__id2cls = {idx:category for category, idx in self.__cls2id.items()}
    # 模型下采样倍数
    self.stride = 32
    # amp模式下梯度缩放
    self.scaler = torch.cuda.amp.GradScaler()
    
    
  def build_model(self):
    self.model = YoLoNet(args.backbone, args.boxes, args.num_classes, args.backbone_pretrained)
    self.model.to(self.device)
    
    
  def create_dataloader(self):
    self.dataset = YoLoDataSet(args.trainFolder, args.img_size, 
                               batchsize=args.batchsize, 
                               multiscale=eval(args.multiscale))
    self.dataloader = data.DataLoader(self.dataset, 
                                      batch_size=args.batchsize,
                                      shuffle=True,
                                      num_workers=4,
                                      pin_memory=True
                                      )
    
    
  def warmUp(self, epoch):
    # 学习率预热
    ratio = args.lr / args.warmup_epochs   
    lr = epoch * ratio
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = lr
      print("Adjust Learning Rate:%.7f"%lr)
      
      
  def learningRateDecay(self, lr):
    # 学习率衰减
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = lr
      print("Adjust Learning Rate:%.7f"%lr)
      
      
  def train(self, epoch):
    self.model.train()
    # 学习率调整
    if epoch <= args.warmup_epochs:
      self.warmUp(epoch)
    if epoch == 80:
      self.learningRateDecay(args.lr * 0.1)
    elif epoch == 120:
      self.learningRateDecay(args.lr * 0.01)
    elif epoch == 140:
      self.learningRateDecay(args.lr * 0.001)
      
    for imgs, labs, mask in self.dataloader:
      imgs = imgs.to(self.device)
      labs = labs.to(self.device)
      mask = mask.to(self.device)
      labs = labs.permute([0,3,1,2])
      
      # with torch.cuda.amp.autocast():
      # forward
      # Nx(5+7)x7x7
      preds = self.model(imgs)
      # # 中心点损失
      # center_pos = torch.sum((preds[:,4,:,:] - labs[:,4,:,:])**2*mask)
      # center_neg = torch.sum((preds[:,4,:,:] - labs[:,4,:,:])**2*(1.0-mask))
      # center_loss = center_pos + 0.1 * center_neg
      
      # 注意.clone()开辟新的内存空间.detach()从计算图中剥离
      # -------------注意labs也要.clone(), 否则会原地修改labs中的值-------------- #
      iou = self.calculate_iou(preds[:, :4, :, :].clone().detach(), labs[:, :4, :, :].clone())
      center_pos = torch.sum((preds[:,4,:,:] - iou)**2*mask)
      center_neg = torch.sum((preds[:,4,:,:] - iou)**2*(1.0 - mask))
      center_loss = center_pos + 0.5 * center_neg
      
      # 类别概率损失
      cls_loss = torch.sum((preds[:,5:,:,:]*preds[:,4:5,:,:] - labs[:,5:,:,:])**2*mask.unsqueeze(1))
      # 关键点偏移量损失
      offset_loss = torch.sum((preds[:,:2,:,:] - labs[:,:2,:,:])**2*mask.unsqueeze(1))
      # 边界框尺度损失
      scale_loss = torch.sum((torch.sqrt(preds[:,2:4,:,:]+1e-8) \
                              - torch.sqrt(labs[:,2:4,:,:])+1e-8)**2*mask.unsqueeze(1))
      bbox_loss = offset_loss + scale_loss
      # loss在batch维度取平均
      loss = (5.0*bbox_loss + cls_loss + center_loss) / imgs.size(0)
        
      # backward
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      # # 混合精度训练
      # self.scaler.scale(loss).backward()
      # self.scaler.step(self.optimizer)
      # self.scaler.update()
      scale = self.scaler.get_scale()
      
      # print(imgs.shape, loss.item())
      self.global_step += 1
      if self.global_step % 10 == 0:
        print("Epoch:[{:03d}]-Loss:{:.3f}-bbox_loss:{:.3f}-cls_loss:{:.3f}-center_loss:{:.3f}-scale:{:.3f}"\
              .format(epoch, loss.item(), bbox_loss.item(), cls_loss.item(), center_loss.item(), scale))
      
      if self.global_step % 100 == 0:
        self.display(imgs.detach().cpu(), preds.detach().cpu())
        
    # epoch finish evaluate
    # save
    torch.save(self.model.state_dict(), os.path.join(args.checkpoint, "epoch_%d.pt"%epoch))
    
    
  def evaluate(self):
    # TODO评测
    pass
  
  
  def xywh2xyxy(self, bboxes):
    # 转换bbox: xywh格式到xyxy格式, shape: Nx4x7x7
    # 转换的尺度:在feature map尺度上[e.g. 7x7]
    grid_size = bboxes.size(-1)
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
    grid_y = grid_y.to(self.device)
    grid_x = grid_x.to(self.device)
    # print(bboxes)
    # 转换中心点坐标
    bboxes[:, 0, :, :] = bboxes[:, 0, :, :] + grid_x.unsqueeze(0)
    bboxes[:, 1, :, :] = bboxes[:, 1, :, :] + grid_y.unsqueeze(0)
    bboxes[:, 2, :, :] = bboxes[:, 2, :, :] * grid_size
    bboxes[:, 3, :, :] = bboxes[:, 3, :, :] * grid_size
    # convert to xyxy format
    # 由于torch计算梯度的原因,此处不能原地修改
    bboxes[:, 0, :, :] = bboxes[:, 0, :, :] - 0.5 * bboxes[:, 2, :, :]
    bboxes[:, 1, :, :] = bboxes[:, 1, :, :] - 0.5 * bboxes[:, 3, :, :]
    bboxes[:, 2, :, :] = bboxes[:, 0, :, :] + bboxes[:, 2, :, :]
    bboxes[:, 3, :, :] = bboxes[:, 1, :, :] + bboxes[:, 3, :, :]
    return bboxes
    
    
  def calculate_iou(self, bboxes1, bboxes2):
    # 计算预测的bboxes和gt bboxes之间的iou值作为中心点置信度损失
    # 中心点置信度损失不仅要衡量是否包含目标, 还要衡量目标预测的好坏
    # Nx4x7x7
    bboxes1 = self.xywh2xyxy(bboxes1)
    bboxes2 = self.xywh2xyxy(bboxes2)
    # print(bboxes1)
    # compute iou
    bboxes1_area = (bboxes1[:, 2, :, :] - bboxes1[:, 0, :, :]) * (bboxes1[:, 3, :, :] - bboxes1[:, 1, :, :])
    bboxes1_area.clamp_min_(0.0)
    bboxes2_area = (bboxes2[:, 2, :, :] - bboxes2[:, 0, :, :]) * (bboxes2[:, 3, :, :] - bboxes2[:, 1, :, :])
    bboxes2_area.clamp_min_(0.0)
    # 计算交集
    inter_box1 = torch.maximum(bboxes1[:, :2], bboxes2[:, :2])
    inter_box2 = torch.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_area = (inter_box2[:, 0] - inter_box1[:, 0]).clamp_min(0.0) * (inter_box2[:, 1] - inter_box1[:, 1]).clamp_min(0.0)
    iou = inter_area / (bboxes1_area + bboxes2_area - inter_area + 1e-8)
    # Nx7x7
    return iou
  
  
  def display(self, imgs, preds):
    # 可视化预测结果
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # restore imgs
    # 1x3x1x1
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imgs = imgs * std + mean
    # convert HxWxC
    imgs = imgs.permute([0,2,3,1]).contiguous()
    imgs = (255.0 * imgs).numpy().astype(np.uint8)
    bboxes = []
    for i in range(imgs.shape[0]):
      box = []
      # 得到中心点目标
      grid_y, grid_x = torch.where(preds[i,4,:,:] >= 0.5)
      if len(grid_y) == 0:
        bboxes.append(box)
        continue
      # 得到中心点偏移量
      offset_x, offset_y = preds[i,0,grid_y,grid_x], preds[i,1,grid_y,grid_x]
      # 得到尺度
      width = preds[i,2,grid_y,grid_x] * args.img_size
      height = preds[i,3,grid_y,grid_x] * args.img_size
      # 得到前景概率
      confidence = preds[i,4,grid_y,grid_x]
      # 得到目标类别概率
      cls_prob = (confidence * preds[i,5:,grid_y,grid_x]).t()
      c_x = (grid_x + offset_x) * 32
      c_y = (grid_y + offset_y) * 32
      cls_idx = torch.argmax(cls_prob, dim=1).numpy()
      # xmin = (c_x - 0.5 * width).int().clamp_min(0).numpy()
      # xmax = (c_x + 0.5 * width).int().clamp_max(224).numpy()
      # ymin = (c_y - 0.5 * height).int().clamp_min(0).numpy()
      # ymax = (c_y + 0.5 * height).int().clamp_max(224).numpy()
      # -------------------------------------------- #
      # bbox可视化格式是:[x,y,w,h,c]
      for x, y, w, h, idx in zip(c_x.numpy(), c_y.numpy(), width.numpy(), height.numpy(), cls_idx):
        bnd = [x, y, w, h, self.__id2cls[idx]]
        box.append(bnd)
      bboxes.append(box)
    for i, (img, box) in enumerate(zip(imgs, bboxes)):
      display_detects(img.copy(), box, os.path.join(self.imgLogFolder, "test_%d.jpg"%i))
      
    
  def main(self):
    for epoch in range(1, args.epochs+1):
      self.train(epoch)
    # 保存权重到dbfs
    os.system("cp %s %s"%(os.path.join(args.checkpoint, "epoch_%d.pt"%epoch)),
              "/dbfs/mnt/algo-data/weishao/lucky_yolo/epoch_multiscale.pt")
      
      
      
if __name__ == "__main__":
  yoloTrainer = LuckyYoLoTrainer()
  yoloTrainer.main()
    