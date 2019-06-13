
# coding: utf-8

# In[68]:


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

# Set some parameters
IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_CHANNELS = 3
TRAIN_PATH = './dataset/train_data/'
TEST_PATH = './dataset/test_data/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed





# In[69]:


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
print(train_ids)
print(test_ids)


# In[108]:


train_images = next(os.walk(TRAIN_PATH+train_ids[0]+"/"))[2]
train_masks = next(os.walk(TRAIN_PATH+train_ids[2]+"/"))[2]
test_images = next(os.walk(TEST_PATH+test_ids[0]+"/"))[2]
X_train = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),dtype = np.uint8)

Y_train = np.zeros((len(train_masks), IMG_HEIGHT, IMG_WIDTH, 1),dtype=np.bool)
print(len(X_train))
print(len(Y_train))
print('getting and resizing train images and masks...')
count_i=0
for images in train_images:
    path = TRAIN_PATH + train_ids[0]+"/"+ images
    img = imread(path)[:,:,:IMG_CHANNELS]
    #img = resize(img, (IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    X_train[count_i] = img
    count_i = count_i + 1
count_m=0
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for mask_file in train_masks:
    path = TRAIN_PATH+train_ids[2]+"/" + mask_file
    mask_ = imread(path)
    ret, mask_ = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY_INV)
    mask_=mask_[:,:,np.newaxis]
    Y_train[count_m] = mask_
    count_m = count_m + 1

X_test = np.zeros((len(test_images),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
sizes_test=[]
print('Getting and resizing test images ... ')
count_t=0
for test in test_images:
    path = TEST_PATH+test_ids[0]+"/"+test
    
    img = imread(path)[:,:,:IMG_CHANNELS]
    X_test[count_t] = img
    count_t=count_t+1
print('Done!')


# In[109]:


# Check if training data looks all right
ix = random.randint(0, len(train_images))
print(X_train[ix].size)
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


# In[116]:


def mean_iou(y_true,y_pred):
    prec = []
    for t in np.arange(0.5,1.0,0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_,2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec),axis=0)


# In[117]:


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

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

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# In[118]:


earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('tumor_test.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])


# In[120]:


# Predict on train, val and test
model = load_model('tumor_test.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (96, 96), 
                                       mode='constant', preserve_range=True))


# In[135]:


ix = random.randint(0, len(preds_test_t))
imshow(X_test[0])
plt.show()
imshow(np.squeeze(preds_test_upsampled[0]))
plt.show()

