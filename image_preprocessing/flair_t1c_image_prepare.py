
## 프로젝트 명       : brain_tumor_semantic_segmentation
## 프로젝트 목적     : 이미지를 전처리하여 뇌의 병변부분을 찾아내서 mask로 만들고 그것들을 이용해 semantic_segmentation을 한다.
## 스크립트 기능     : 이미지 전처리
## 스크립트 명       : flair_t1c_image_prepare.py
##                 1. imagepre = 형태학적 이미지 전처리를 통해 병변추출
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
class flair_t1c_image_prepare:
    def imagepre(img1):
   

        kernel = np.ones((3,3),np.uint8)                                #(3,3)커널 만들기
        gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)   #모폴로지 연산
        median = cv2.medianBlur(img1,5)                                 #미디언 필터를 이용한 블러처리( 잡음 제거 )
        img=cv2.add(gradient,median)                                    #두이미지를 더한다.

        #### 전처리 이미지 생성################################
        meaning=np.mean(img)                                            #전처리 이미지의 평균값
        stand=np.std(img)                                               #전처리 이미지의 표준편차
        meaning2=np.mean(gradient)                                      #형태학적 기울기의 평균값
        thre = meaning+stand-meaning2                                   #연산을 통해 임계값을 구한다.
        
        ret,img_result1= cv2.threshold(img,thre+30,255,cv2.THRESH_BINARY_INV)   #flair영상을 넣었을 때 위의 임계값으로는 완벽하게 병변도출이 되지않음 그래서 30을 더해서 더나은 mask만듬

        ################i2 생성#############################

        closing = cv2.morphologyEx(img_result1,cv2.MORPH_CLOSE,kernel)          #close
        opening = cv2.morphologyEx(img_result1,cv2.MORPH_OPEN,kernel)           #opening
        thresh= cv2.add(closing,opening)                                        #전처리를 통해 계선된 이미지 얻음

        ###############i3 이미지##################################


        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)                                        #1번 전처리한 원본이미지를 그레이스케일로
        im_th = thresh.copy()                                                               #i3이미지를 복사해옴
        imgray1 = cv2.cvtColor(im_th,cv2.COLOR_BGR2GRAY)                                    
        ret , thr =cv2.threshold(imgray1, 127, 255 ,0)                                      
        contours,hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #thr에서 잡티를 포함한 병변부분이라 예상되는 부분의 테두리를 얻는다.
        h, w =im_th.shape[:2]                                                               #원본의 높이와 넓이 받아온다
        mask = np.zeros((h +2,w+2),np.uint8)                                    
        im_t1 =list()

        for i in range(0,len(contours)):                                                    # contours중 가장 큰 크기의 contours를 찾아내는 알고리즘
                cnt = contours[i]

                im_t1.append(len(cnt))
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_t2=max(im_t1)
        ################################################################################max_contours
        new_cnt =list()
        for cont in contours:
            if len(cont) == im_t2 :                                                         #contours중에 가장 큰 contour를 찾는다.
                for cnt in cont :                                                           
                    
                    x=list(itertools.chain(*cnt))                                           #차원감소를 해주는 부분
                    new_cnt.append(x)                                                       #new_cnt배열에 추가해준다.

        for cnt in new_cnt:                                                                 #병변이 있을것같은 위치를  픽셀값을 200으로 바꾼다.
            
            im_th.itemset(cnt[1],cnt[0],0,200)
            im_th.itemset(cnt[1],cnt[0],1,200)
            im_th.itemset(cnt[1],cnt[0],2,200)
        imgray = cv2.cvtColor(im_th,cv2.COLOR_BGR2GRAY)

        cv2.floodFill(imgray,mask ,(0,0),(255),210,210)                                     #200인곳을 제외하고 나머지는 0으로 바꾼다
        img_test=cv2.bitwise_xor(imgray,thr)
        img_test=cv2.bitwise_and(img1,img_test)
        img_test = cv2.morphologyEx(img_test,cv2.MORPH_CLOSE,kernel)                        #원본과 floodfill한 이미지를 합치고 close연산으로 다듬어준다.

        return img_test                                                                     #이미지를 리턴한다.


# In[58]:




