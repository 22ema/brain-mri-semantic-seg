
# coding: utf-8

# In[59]:


import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import itertools
class flair_t1c_image_prepare:
    def imagepre(img1):
   

        kernel = np.ones((3,3),np.uint8)## 커널 만들기w
        gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)##모폴로지 연산
        median = cv2.medianBlur(img1,5)
        img=cv2.add(gradient,median)

        #### 전처리 이미지 생성################################



        meaning=np.mean(img)##
        stand=np.std(img)
        meaning2=np.mean(gradient)
        thre = meaning+stand-meaning2
        
        ret,img_result1= cv2.threshold(img,thre+30,255,cv2.THRESH_BINARY_INV)

        ################i2 생성#############################

        closing = cv2.morphologyEx(img_result1,cv2.MORPH_CLOSE,kernel)
        opening = cv2.morphologyEx(img_result1,cv2.MORPH_OPEN,kernel)
        thresh= cv2.add(closing,opening)

        ###############i3 이미지##################################


        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        im_th = thresh.copy()
        imgray1 = cv2.cvtColor(im_th,cv2.COLOR_BGR2GRAY)
        ret , thr =cv2.threshold(imgray1, 127, 255 ,0)
        contours,hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w =im_th.shape[:2]
        mask = np.zeros((h +2,w+2),np.uint8)
        im_t1 =list()

        for i in range(0,len(contours)):
                cnt = contours[i]

                im_t1.append(len(cnt))
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_t2=max(im_t1)
        new_cnt =list()
        for cont in contours:
            if len(cont) == im_t2 :
                for cnt in cont :
                    
                    x=list(itertools.chain(*cnt))
                    new_cnt.append(x)

        for cnt in new_cnt:
            
            im_th.itemset(cnt[1],cnt[0],0,200)
            im_th.itemset(cnt[1],cnt[0],1,200)
            im_th.itemset(cnt[1],cnt[0],2,200)
        imgray = cv2.cvtColor(im_th,cv2.COLOR_BGR2GRAY)

        cv2.floodFill(imgray,mask ,(0,0),(255),210,210)
        img_test=cv2.bitwise_xor(imgray,thr)
        img_test=cv2.bitwise_and(img1,img_test)
        img_test = cv2.morphologyEx(img_test,cv2.MORPH_CLOSE,kernel)

        return img_test


# In[58]:




