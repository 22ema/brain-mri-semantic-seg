## 프로젝트 명       : brain_tumor_semantic_segmentation
## 프로젝트 목적     : 이미지를 전처리하여 뇌의 병변부분을 찾아내서 mask로 만들고 그것들을 이용해 semantic_segmentation을 한다.
## 스크립트 기능     : semantic_segmantation
## 스크립트 명       : mri_segmentaition.py
##                 1. mean_iou = 정확도 대신 IOU를 결과로 표시하게 해주는 함수
## 작성 시작 일시    : 2019. 06. 12.
## 마지막 수정 일시  : 2019. 06. 14.
## 작성자 및 수정자  : 한윤성
## e-mail         : yunsung9503@gmail.com 
## github 링크      : https://github.com/22ema/brain-mri-semantic-seg

import os
import sys
import random
import warnings
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# 파라미터들 셋팅하기
IMG_WIDTH    = 96                                                              #image_width
IMG_HEIGHT   = 96                                                              #image_height
IMG_CHANNELS = 3                                                               #image_channel
TRAIN_PATH   = './dataset/train_data/'                                         #train_data_path
TEST_PATH    = './dataset/test_data/'                                          #test_data_path

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# random value setting
seed            = 42
random.seed     = seed
np.random.seed  = seed



train_ids = next(os.walk(TRAIN_PATH))[1]                                    #이미지 경로안의 폴더들을 배열의 형태로 받아온다.
test_ids  = next(os.walk(TEST_PATH))[1]                                     #이하동문(test데이터이다.!)
print(train_ids)                                                            #확인 작업 없어도됨.
print(test_ids)



train_images = next(os.walk(TRAIN_PATH+train_ids[0]+"/"))[2]                                        #학습이미지(원본)을 들고오는 부분이다.
train_masks  = next(os.walk(TRAIN_PATH+train_ids[2]+"/"))[2]                                        #학습이미지(마스크,즉 전처리과정에서 얻어진 병변 이미지)를 들고오는 부분.
test_images  = next(os.walk(TEST_PATH+test_ids[0]+"/"))[2]                                          #테스트 이미지를 들고오는 부분 이때 이이미지들은 학습할 때 절때 사용하지 않는다.
X_train      = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),dtype = np.uint8)  #학습이미지를 저장할 4차원 numpy배열을 만든다.

Y_train = np.zeros((len(train_masks), IMG_HEIGHT, IMG_WIDTH, 1),dtype=np.bool)                      #마스크 이미지즉 정답이미지들을 저장할 4차원 numpy배열을 만든다. 
                                                                                                    #이때 이미지는bool형태로 저장한다. 참거짓으로 판단하기 때문이다.
print(len(X_train))                                                                                 #여기도 없어도된다 그냥 확인하기위해 print해보았다.
print(len(Y_train))
print('getting and resizing train images and masks...')

count_i = 0                                                                                         #이미지의 개수를 파악해서 x_train배열에 담기위해 지정
for images in train_images:                                                                         #train_images 배열안에서 차례대로 값을 꺼내와 images에 넣어서 배열의 끝까지 반복함
    path = TRAIN_PATH + train_ids[0]+"/"+ images                                                    #images의 폴더안의 위치
    img = imread(path)[:,:,:IMG_CHANNELS]                                                           #이미지를 읽어와서 3채널값으로 바꿔주는 작업 이유는 망안에서 3채널로 입력받음
    #img = resize(img, (IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)                  #이미 사이즈를 바꾸는 알고리즘은 만들어서 따로 바꿈
    X_train[count_i] = img                                                                          #x_train배열에 저장
    count_i = count_i + 1                                                                           #이미지의 개수만큼 배열의 window를 이동시켜주기 위해

count_m = 0
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for mask_file in train_masks:                                                                       #mask를 만드는 과정 원본이미지를 저장하는것과 동일하다. 다른부분은 
                                                                                                    #채널값을 1로 지정한것 이다.
    path = TRAIN_PATH+train_ids[2]+"/" + mask_file
    mask_ = imread(path)
    ret, mask_ = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY_INV)
    mask_=mask_[:,:,np.newaxis]
    Y_train[count_m] = mask_
    count_m = count_m + 1

X_test      = np.zeros((len(test_images),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
sizes_test  =[]
print('Getting and resizing test images ... ')                                                     #test이미지도 위의 이미지 저장방식과 동일하다. 위의 것을 참고하면된다.
count_t=0
for test in test_images:
    path = TEST_PATH+test_ids[0]+"/"+test
    
    img = imread(path)[:,:,:IMG_CHANNELS]
    X_test[count_t] = img
    count_t=count_t+1
print('Done!')

def mean_iou(y_true,y_pred):                                                                        #matrics에서 정확도 말고 IOU수치를 쓰기위해 설정하는 부분. 
    prec = []
    for t in np.arange(0.5,1.0,0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_,2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec),axis=0)




inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)                                                          #input의 값을 모두 255으로 나눠주는 람다문이다.
## 이밑부터 unet 모델입니다.##################################################################################################################
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
###############################################################################################
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])                 #optimizer = adam 분류 문제에 탁월한 crossentropy 이용
model.summary()



earlystopper = EarlyStopping(patience=5, verbose=1)                                             #같은 IOU가 5번정도 나오면 멈춤
checkpointer = ModelCheckpoint('tumor_test.h5', verbose=1, save_best_only=True)                 #각 epoch마다 h5파일의 형태로 저장하여 덮어 씌운다.
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,           #결과 저장
                    callbacks=[earlystopper, checkpointer])




# test를 하기위해 test영상을 불러온다
model = load_model('tumor_test.h5', custom_objects={'mean_iou': mean_iou})                      #모델을 불러온다
preds_test = model.predict(X_test, verbose=1)                                                   #X_test배열의 이미지들을 모델에 넣어서 테스트한다.

# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)                                              #임계값 설정 0.5이하이면 틀린값

# Create list of upsampled test masks
preds_test_upsampled = []                                                                       #결과 이미지들을 저장하는 list
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (96, 96), 
                                       mode='constant', preserve_range=True))





ix = random.randint(0, len(preds_test_t))                                                       #랜덤이미지를 부르기 위해
imshow(X_test[ix])                                                                               #원본이미지 부르기
plt.show()
imshow(np.squeeze(preds_test_upsampled[ix]))                                                     #결과 이미지들 출력
plt.show()

