
'''
Same code but implemented on cocodataset
'''

import tensorflow as tf
import tensornets as nets
from cocobuilder import gp_builder,imobj, categ_switcher
import numpy as np
import matplotlib.pyplot as plt
import math
import coco
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


data_dir = '/lfs02/datasets/coco/'
ann_dir  = '/home/alex054u4/data/nutshell/coco/annotations'

#Data Stuff
builder = gp_builder('coco')
builder.set_data()
train_dataset= builder._get_data()

# Define the model hyper parameters
N_classes=80
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolo=nets.YOLOv2(x, nets.MobileNet50v2, is_training=True, classes=N_classes)

step = tf.Variable(0,trainable=False)
#Optimizer
lr       = tf.Variable(1e-3,trainable=False)
train    = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(yolo.loss)


#Check points
checkpoint_path   = "/home/alex054u4/data/nutshell/training_trial-COCO"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)

init_op     = tf.global_variables_initializer()
train_saver = tf.train.Saver(max_to_keep=2)


def evaluate_accuracy(data_type='tr'):
  if (data_type  == 'tr'): acc_data  = coco.load(data_dir,ann_dir,'train2017',total_num =48)
  elif(data_type == 'te') : acc_data  = coco.load(coco_dir,ann_dir, 'val2017', total_num=48)
  
  results = []
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolo, {x: yolo.preprocess(img),is_training: False})
    boxes=yolo.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
  if (data_type  =='tr'):return coco.evaluate(results, ann_dir, 'train2017')
  elif (data_type=='te'):return coco.evaluate(results, ann_dir, 'val2017')


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

    trains = builder.load_train(train_dataset,classes=80,batch_size = 64)
    
    pbar = tqdm(total = 1848) 
    for (imgs, metas) in trains:
      # `trains` returns None when it covers the full batch once
      if imgs is None:break

      metas.insert(0, yolo.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolo.loss],dict(zip(yolo.inputs, metas)))
      losses.append(outs[-1])
      pbar.update(1)

    pbar.close()
    print_out='epoch:'+str(i)+'lr: '+str(lr.eval())+ 'loss:'+str(losses[-1])
    print(print_out)
    print(evaluate_accuracy('tr'))
    print('/n/n')
    print(evaluate_accuracy('te'))
    sess.run(step.assign(i))

    train_saver.save(sess,checkpoint_prefix)
