import json
import numpy as np
import cv2


import sys
args = sys.argv



# jsonのロード
with open(args[1], 'r') as f:
    data = json.load(f)

# 黒背景画像の用意
img_ = cv2.imread('original.jpg')
img = np.zeros_like(img_)

# 関節毎の描画色
colors = [(255.,     0.,    85.), (255.,     0.,     0.), (255.,    85.,     0.), (255.,   170.,     0.), (255.,   255.,     0.), (170.,   255.,     0.), (85.,   255.,     0.), (0.,   255.,     0.), (0.,   255.,    85.), (0.,   255.,   170.), (0.,   255.,   255.), (0.,   170.,   255.), (0.,    85.,   255.), (0.,     0.,   255.), (255.,     0.,   170.), (170.,     0.,   255.), (255.,     0.,   255.), (85.,     0.,   255.)]

# 検出された全員について
for d in data['people']:
     kpt = np.array(d['pose_keypoints_2d']).reshape((18, 3))
     # 関節位置の全ペアについて
     for p in pairs:
         pt1 = tuple(list(map(int, kpt[p[0], 0:2])))
         c1 = kpt[p[0], 2]
         pt2 = tuple(list(map(int, kpt[p[1], 0:2])))
         c2 = kpt[p[1], 2]
         # 信頼度0.0の関節は無視
         if c1 == 0.0 or c2 == 0.0:
             continue
         # 関節の描画
         color = tuple(list(map(int, colors[p[0]])))
         img = cv2.line(img, pt1, pt2, color, 7)

cv2.imshow('nacho', img)
