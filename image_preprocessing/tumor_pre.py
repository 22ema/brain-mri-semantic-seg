
# coding: utf-8

# In[12]:


import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import itertools
import os
from flair_t1c_image_prepare import flair_t1c_image_prepare

###이미지 불러오기
new_path = './image'
count= 0
file_list=list()
for image,_,files in os.walk(new_path):
    file_list = sorted(files)

## 이미지 binary_image 로 만들어 gray_data폴더에 저장
for file in file_list:
    img = cv2.imread(new_path+file,cv2.IMREAD_COLOR)
    img=flair_t1c_image_prepare.imagepre(img)
    cv2.imwrite("test_t1c/"+file,img)
print("---------finish------")
    

