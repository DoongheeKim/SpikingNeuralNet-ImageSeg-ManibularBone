#!/usr/bin/env python
# coding: utf-8

#
# Setup
#

import tensorflow as tf
print('tf.__version__=', tf.__version__)

''' #for using GPU on google colab
#Select GPU 0 or 1
GPU_INDEX = 0 #1
GPU_MEM = 0.95
#GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = '/gpu:'+ str(GPU_INDEX)
#Set Memory limit (DO NOT OVER 5G!!!)
tf.config.experimental.set_virtual_device_configuration(gpus[GPU_INDEX], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEM*10*1024)])
'''

''' #for google colab
!pip install keras_spiking
'''

import matplotlib.pyplot as plt
import numpy as np
import keras_spiking
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

import os
import sys
import random
import warnings

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imsave, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras.callbacks import EarlyStopping, ModelCheckpoint

from measure import f_f1_score

''' #for google colab
from google.colab import drive
drive.mount('/content/drive')
#drive.flush_and_unmount()
'''

#
# Parameters
#
#the size of raw images is 3100 x 1300 pixels
IMG_WD= 128 #64  256
IMG_HT= 64 #32 124

IMG_CHANNELS = 1
MASK_CHANNELS = 1
n_steps = 120 #60

DENT_FILE_PATH = "DentalPanoramicXrays"

''' #for google colab
DENT_FILE_PATH = '/content/drive/MyDrive/Colab Notebooks/DentalPanoramicXrays'
'''
IMAGE_FOLDER = "Images"
MASK_FOLDER = "Segmentation1"

image_path = DENT_FILE_PATH + "/" + IMAGE_FOLDER
mask_path = DENT_FILE_PATH + "/" + MASK_FOLDER
#print("image_path=", image_path)
#print("mask_path=", mask_path)

#
# Reading and resizing image files and mask files
#
def read_data(n_read_imgs, image_path, mask_path) :
  image_train = np.zeros((n_read_imgs, IMG_HT, IMG_WD, IMG_CHANNELS), dtype=np.uint8)
  mask_train = np.zeros((n_read_imgs, IMG_HT, IMG_WD, MASK_CHANNELS), dtype=np.uint8)

  sys.stdout.flush()
  n_read = 0
  for (root, dirs, files) in os.walk(image_path):
    for id, file_name in enumerate(files):
       img = imread(image_path + '/' + file_name)[:,:]

       img = resize(img, (IMG_HT, IMG_WD), mode='constant', preserve_range=True)
       img = img[..., np.newaxis]

       image_train[id]= img
       n_read = n_read + 1
       if(n_read >= n_read_imgs) :
          break
       
  print("Reading and resizing image files is done!")
  print()

  sys.stdout.flush()  
  n_read = 0
  for (root, dirs, files) in os.walk(mask_path):
    for id, file_name in enumerate(files):
       mask = imread(mask_path + '/' + file_name)[:,:]
       mask = resize(mask, (IMG_HT, IMG_WD), mode='constant', preserve_range=True)
       mask = mask[..., np.newaxis]

       mask_train[id]= mask
       n_read = n_read + 1
       if(n_read >= n_read_imgs) :
          break

  print("Reading and resizing mask files is done!")
  print()

  #binarize mask_train
  mask_threshold = 0
  mask_train[mask_train > mask_threshold]  = 1
  mask_train[mask_train <= mask_threshold]  = 0

  #print("image_train.shape=", image_train.shape)
  #print("mask_train.shape=", mask_train.shape)
  #print()
  #exit()

  n_train_imgs = int(n_read_imgs/2)
  image_train_sequ = image_train[:n_train_imgs]
  mask_train_sequ = mask_train[:n_train_imgs]
  image_train_sequ = np.tile(
    image_train_sequ[:, None], 
    (1, n_steps, 1, 1, 1))

  image_test_sequ = image_train[n_train_imgs:]
  mask_test_sequ = mask_train[n_train_imgs:]
  image_test_sequ = np.tile(image_test_sequ[:, None], (1, n_steps, 1, 1, 1))


  image_train = []
  mask_train = []
  return image_train_sequ, mask_train_sequ, image_test_sequ, mask_test_sequ

def display(d_list, title):
  plt.figure(figsize=(15, 15))
  #title = ["Input Image", "True Mask", "Predicted Mask"]

  for i in range(len(d_list)):
    plt.axis("off")
    plt.subplot(1, len(d_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(d_list[i])) 
  plt.show()


#n_read_imgs = 4
n_read_imgs = len( next(os.walk(image_path))[2] )
image_train_seq, mask_train_seq, image_test_seq, mask_test_seq = \
  read_data(n_read_imgs, image_path, mask_path) 
  

#Checking data read above
#print an image and its mask randomly chosen.
random_index = np.random.choice(image_train_seq.shape[0])
print("random_idx=", random_index)
sample_image, sample_mask = image_train_seq[random_index][0], mask_train_seq[random_index]
display([sample_image, sample_mask], ["sample_image","sample_mask"])

#normalize image_train_seq to range between 0 and 1 ?")
if(image_train_seq.max() > 1) :
  image_train_seq = image_train_seq/255
  image_test_seq = image_test_seq/255

#print("image_train_seq max=", image_train_seq.max())
#print("image_train_seq min=", image_train_seq.min())

'''
### print 0-th components of all pixels of the middle row of sample_image 
print("print 0-th components of all pixels of the middle row of sample_image")
for x in range(IMG_WD):
  print(sample_image[int(IMG_HT/2)][x][0], end = " ")
  if(x % 30 == 0) :
     print("\n")
print("\n")
#input("go?")

### print 0-th components of all pixels of the middle row of sample_mask 
print("print 0-th components of all pixels of the middle row of sample_mask")
for x in range(IMG_WD):
  print(sample_mask[int(IMG_HT /2)][x][0], end = " ")
  if(x % 32 == 0) :
     print("\n")
print("\n")
#input("go?")
'''


#
# Spiking U-NET model
#
def double_conv(z, n_filters, dt, sp_aware):
    #Conv2D then ReLU activation
    z = tf.keras.layers.TimeDistributed(layers.Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal"))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=sp_aware)(z)
    
    #Conv2D then ReLU activation
    z = tf.keras.layers.TimeDistributed(layers.Conv2D(n_filters, 3, padding = "same",  kernel_initializer = "he_normal"))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=sp_aware)(z)

    return z

def downsample(z, n_filters, dt, sp_aware):
    z1 = double_conv(z, n_filters, dt, sp_aware)
    z2=tf.keras.layers.TimeDistributed(layers.MaxPool2D(2))(z1)
    z2 = tf.keras.layers.TimeDistributed(layers.Dropout(0.3))(z2)
    return z1, z2

def upsample(z, conv_features, n_filters, dt, sp_aware):
    #upsample
    z = tf.keras.layers.TimeDistributed(layers.Conv2DTranspose(n_filters, 3, 2, padding="same"))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=sp_aware)(z)
 
    #concatenate 
    z = layers.concatenate([z, conv_features])

    #dropout
    z = tf.keras.layers.TimeDistributed(layers.Dropout(0.3))(z)

    #Conv2D twice with ReLU activation
    z = double_conv(z, n_filters, dt, sp_aware)

    return z

def build_unet_model(n_steps, dt, sp_aware):
    # inputs
    inputs = layers.Input(shape=(n_steps, IMG_HT, IMG_WD, IMG_CHANNELS))

    #encoder:  downsample
    #1 - downsample
    f1, g1 = downsample(inputs, 64, dt, sp_aware)
    #2 - downsample
    f2, g2 = downsample(g1, 128, dt, sp_aware)
    #3 - downsample
    f3, g3 = downsample(g2, 256, dt, sp_aware)
    #4 - downsample
    f4, g4 = downsample(g3, 512, dt, sp_aware)

    #5 - bottleneck
    d_c = double_conv(g4, 1024, dt, sp_aware)

    #decoder:  upsample
    #6 - upsample
    up6 = upsample(d_c, f4, 512, dt, sp_aware)
    #7 - upsample
    up7 = upsample(up6, f3, 256, dt, sp_aware)
    #8 - upsample
    up8 = upsample(up7, f2, 128, dt, sp_aware)
    #9 - upsample
    up9 = upsample(up8, f1, 64, dt, sp_aware)

    #outputs
    print("up9 shaoe before mean=", up9.shape)
    up9 = tf.reduce_mean(up9, 1)
    print("u9 shaoe after mean=", up9.shape)

    outputs = layers.Conv2D(2, 1, padding="same", activation = "softmax")(up9)

    #unet model using Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


#
#Training
#

dt = 0.2
N_EPOCHS = 50  #3
BATCH_SIZE = 1#

''' #for gpu on google colab
with tf.device(gpu): 
'''
if(True) :
  spiking_aware_train=True #$False #True
  keras_spiking.default.dt = dt  # default 0.01

  unet_model = build_unet_model(n_steps, dt, spiking_aware_train)
  unet_model.summary()

''' #for gpu on google colab
with tf.device(gpu):
'''
if(True) :
  unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss="sparse_categorical_crossentropy",
                   metrics="accuracy"
                   )
  

  earlystopper = EarlyStopping(patience=5, verbose=1)
  print('earlystopper=', earlystopper )
  checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
  print('checkpointer=', checkpointer )

  #hitory returns validation callbacks
  history = unet_model.fit(image_train_seq, mask_train_seq,
                               epochs=N_EPOCHS,
                               validation_split=0.5, #TEMP 0.2
                               batch_size = BATCH_SIZE, 
                               #steps_per_epoch=STEPS_PER_EPOCH,
                               #validation_steps=VALIDATION_STEPS,
                               #validation_data=validation_batches
                               )
  
#save the trained model in a h5 file, and can use it later
#unet_model.save("unet_pet_KerasSpiking_model.h5")
#unet_model = tf.keras.models.load_model('unet_pet_KerasSpiking_model.h5')


#
# Test Segmentation 
#
def gen_mask(pred_mask, idx):
  pred_mask = tf.argmax(pred_mask, axis=-1) #axis=-1 means last axis
  pred_mask = pred_mask[..., tf.newaxis]  # ... means "as many ; as"
 
  return pred_mask[idx]

def show_pred(model, sample_image, sample_mask):
    pred = model.predict(sample_image[tf.newaxis,
          ...])
    pred_mask = gen_mask(pred, 0)
    display([sample_image, sample_mask,pred_mask], 
	["image", "mask", "pred_mask"])
  
n_test_imgs = len(image_test_seq)
preds = unet_model.predict(image_test_seq)

#showing predictions and accuracies
acc_sum = 0
acc_f1_s = 0
n_test = 0
for test_idx in range(n_test_imgs) :
  if(input("continue for showing predictionc ? (y or n)") != "y"):
    n_test = test_idx
    break

  pred_mask = gen_mask(preds, test_idx)
  pred_mask = np.asarray(pred_mask)

  etest_image = image_test_seq[test_idx][0]
  etest_mask = mask_test_seq[test_idx]

  display([etest_image, etest_mask], ["etest_image", "etest_mask"])
  print()
  print("test_idx=", test_idx)
  etest_image[pred_mask == 0] = 0
  seg_img = etest_image
  display([pred_mask, seg_img], ["pred_mask", "segmented_test_image"])

  #accuracy calculaion
  m = tf.keras.metrics.Accuracy()
  m.update_state(pred_mask, etest_mask)
  accuracy = m.result().numpy()
  print("accuracy=", accuracy, "at idx=", test_idx)
  acc_sum = acc_sum +  accuracy

  f1_s = f_f1_score(etest_mask, pred_mask)
  print("f1_score=", f1_s, "at idx=", test_idx)
  acc_f1_s = acc_f1_s + f1_s

if(n_test == n_test_imgs) :
  acc_mean = acc_sum / n_test_imgs 
  f1_s_mean = acc_f1_s / n_test_imgs 
else :
  acc_mean = acc_sum / n_test 
  f1_s_mean = acc_f1_s / n_test

print("accuracy mean = ", acc_mean)
print("f1_score mean = ", f1_s_mean)

print("end of test segmentation")

