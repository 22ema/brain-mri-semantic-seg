
## 프로젝트 명       : brain_tumor_semantic_segmentation
## 프로젝트 목적     : 이미지를 전처리하여 뇌의 병변부분을 찾아내서 mask로 만들고 그것들을 이용해 semantic_segmentation을 한다.
## 스크립트 기능     : 이미지 전처리
## 스크립트 명       : tumor_pre.py
##                 1. 이미지를 받아와 flair_image_prepare를 실행한다.
## 작성 시작 일시    : 2019. 06. 12.
## 마지막 수정 일시  : 2019. 06. 14.
## 작성자 및 수정자  : 한윤성
## e-mail         : yunsung9503@gmail.com 
## github 링크      : https://github.com/22ema/brain-mri-semantic-seg

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import itertools
import os
from flair_t1c_image_prepare import flair_t1c_image_prepare

###이미지 불러오기
new_path = './fin_flair/fin_flair_dataset'
count= 0
file_list=list()
for image,_,files in os.walk(new_path):
    file_list = sorted(files)

## 이미지 binary_image 로 만들어 gray_data폴더에 저장
for file in file_list[:5]:
    img = cv2.imread(new_path+file,cv2.IMREAD_COLOR)
    img=flair_t1c_image_prepare.imagepre(img)
    cv2.imwrite("test_flair_test/"+file,img)
print("---------finish------")
    

