from __future__ import division
import yolfnets as nets
import tensorflow as tf
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
import voc
from YOLF_utils import *


voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'

# Define the model hyper parameters
N_classes=20
is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolf=model(x, lmbda=0, dropout_rate=0)


# Define an optimizer
epoch  = tf.Variable(0,trainable=False,name="Epoch")
epo_inc= tf.assign_add(epoch,1,name="Epoch-Update")
#lr     = tf.Variable(1e-4,trainable=False,dtype=tf.float32)
step = tf.Variable(0, name = 'step' ,trainable=False,dtype=tf.int32)
lr = tf.train.piecewise_constant(
    step, [100, 180, 320, 570, 1000, 40000, 60000],
    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-4, 1e-5])
train = tf.train.MomentumOptimizer(lr, 0.9).minimize(yolf.loss,
                                                     global_step=step)

#train  = tf.train.AdamOptimizer(lr, 0.9).minimize(yolf.loss)


#Check points for step training_trial_step
checkpoint_path   = "/home/alex054u4/data/nutshell/training_trial_YOLF_AfterUnderstanding"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)


init_op     = tf.global_variables_initializer()
train_saver = tf.train.Saver(max_to_keep=2)

def evaluate_accuracy(data_type='te'):
  if (data_type  == 'tr'): acc_data  = voc.load(voc_dir % 2012,'trainval')
  elif(data_type == 'te') : acc_data  = voc.load(voc_dir % 2007, 'test')
  
  #print('Train Accuracy: ',voc.evaluate(boxes, voc_dir % 2007, 'trainval'))
  results = []
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolf, {x: yolf.preprocess(img),is_training: False})
    boxes=yolf.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
  if (data_type  =='tr'):return voc.evaluate(results, voc_dir % 2012, 'trainval')
  elif (data_type=='te'):return voc.evaluate(results, voc_dir % 2007, 'test')


with tf.Session() as sess:
  ckpt_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and 'ckpt' in f]
  if (len(ckpt_files)!=0):
    print('CHECK POINTS STORED \n\n')
    train_saver.restore(sess,checkpoint_prefix)
  else:
    sess.run(init_op)


  losses     = np.zeros(344)

  pbar = tqdm(initial=epoch.eval(),total = 233) 
  for i in range(epoch.eval(),233):    
    trains = voc.load_train([voc_dir % 2007, voc_dir % 2012],'trainval', batch_size=48)

    for j,(imgs, metas) in enumerate(trains):
      # `trains` returns None when it covers the full batch once
      if imgs is None: break
      metas.insert(0, yolf.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolf.loss],dict(zip(yolf.inputs, metas)))
      losses[j]=outs[-1]
    

    if(math.isnan(np.mean(losses))):
      file = open('/home/alex054u4/data/Nan_weights.txt','w+')
      for i in tf.trainable_variables():
        file.write(str(i.eval)+'\n')

    #tracc_str,_     = evaluate_accuracy('tr')
    teacc_str,teacc = evaluate_accuracy('te')
    print('\nepoch:',i,'lr: ',lr.eval(),'loss:',np.mean(losses))
    #print('Training Acc',tracc_str)
    print('\n\n')
    print('Dev-Set Acc',teacc)
    print('\n')    
  
    sess.run(epo_inc)
    train_saver.save(sess,checkpoint_prefix)


    #print ('highest training accuacy:', best_acc)
    print ('=================================================================================================================================================================================')
    pbar.update(1)
  pbar.close()
  
