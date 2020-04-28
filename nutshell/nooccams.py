import tensorflow as tf
import tensornets as nets
import voc
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
import random
import cv2
from copy import copy, deepcopy
from pathlib import Path
import os
import time 
from datetime import timedelta
from tqdm import tqdm
#import zipfile
import tarfile
import shutil
import wget
import sys


data_dir = "/home/alex054u4/data/nutshell/newdata/"
voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'

# Define the model hyper parameters
is_training = tf.placeholder(tf.bool)
N_classes=20
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolo=nets.YOLOv2(x, nets.MobileNet50v2, is_training=True, classes=N_classes)
# Define an optimizer
step = tf.Variable(0, trainable=False)
lr = tf.train.piecewise_constant(
    step, [1032,14000,27000,40000,54000,60000,65000,70000,80000],
    [1e-3 , 1e-4,1e-5,1e-4,1e-3,1e-4,1e-5,1e-6, 1e-7, 1e-8])


#Optimizer
train    = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(yolo.loss,global_step=step)


#Check points
checkpoint_path   = "/home/alex054u4/data/nutshell/training_trial5"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)

init_op     = tf.global_variables_initializer()
train_saver = tf.train.Saver(max_to_keep=2)


def evaluate_accuracy(data_type='tr'):
  if (data_type  == 'tr'): acc_data  = voc.load(voc_dir % 2012,'trainval',total_num =48)
  elif(data_type == 'te') : acc_data  = voc.load(voc_dir % 2007, 'test', total_num=48)
  
  #print('Train Accuracy: ',voc.evaluate(boxes, voc_dir % 2007, 'trainval'))
  results = []
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolo, {x: yolo.preprocess(img),is_training: False})
    boxes=yolo.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
  if (data_type  =='tr'):return voc.evaluate(results, voc_dir % 2012, 'trainval')
  elif (data_type=='te'):return voc.evaluate(results, voc_dir % 2007, 'test')


with tf.Session() as sess:
  ckpt_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and 'ckpt' in f]
  if (len(ckpt_files)!=0):
    train_saver.restore(sess,checkpoint_prefix)
  else:
    sess.run(init_op)
    sess.run(yolo.stem.pretrained())

  losses = []
  for i in range(step.eval(),233):
    print(" \n Epoch", i, "starting...")
    # Iterate on VOC07+12 trainval once

    trains = voc.load_train([voc_dir % 2012, voc_dir % 2007],'trainval', batch_size=48)
    
    pbar = tqdm(total = 344) 
    for (imgs, metas) in trains:
      # `trains` returns None when it covers the full batch once
      if imgs is None:break

      metas.insert(0, yolo.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolo.loss],dict(zip(yolo.inputs, metas)))
      losses.append(outs[-1])
      pbar.update(1)

    pbar.close()
    print_out='epoch:'+str(i)+'lr: '+ 'loss:'+str(losses[-1])
    print(print_out)
    print(evaluate_accuracy('tr'))
    print(evaluate_accuracy('te'))

    train_saver.save(sess,checkpoint_prefix)
