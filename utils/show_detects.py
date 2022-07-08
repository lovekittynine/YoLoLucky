#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:15:16 2022

@author: weishaowei
"""

# 可视化检测结果
import cv2
from typing import List



def display_bbox(image, box_list:List):
  for box in box_list:
    x, y, w, h = box
    x1, y1 = int(x - w//2), int(y - h//2)
    x2, y2 = int(x + w//2), int(y + h//2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
  cv2.imwrite("./test_box.jpg", image)
  
  
def display_detects(image, box_list:List, filename:str):
  height, width, _ = image.shape
  for box in box_list:
    if box:
      x, y, w, h, category = box
      x1, y1 = max(0, int(x - w//2)), max(0, int(y - h//2))
      x2, y2 = min(width, int(x + w//2)), min(height, int(y + h//2))
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
      del_w, del_h = min(50, int(width)), min(20, int(height))
      cv2.rectangle(image, (x1, y1), (x1+del_w, y1+del_h), (0, 255, 0), -1, 1)
      cv2.putText(image, category, (x1, y1+10), cv2.FONT_HERSHEY_COMPLEX, 0.38, (0,0,255), 1)
  cv2.imwrite(filename, image)
  
  

if __name__ == "__main__":
  img = cv2.imread("test.jpg")
  h, w, _ = img.shape
  img = cv2.resize(img, (224, 224))
  boxes = [[360*224/w, 187.5*224/h, 240*224/w, 375*224/h, "person"]]
  display_detects(img, boxes)
