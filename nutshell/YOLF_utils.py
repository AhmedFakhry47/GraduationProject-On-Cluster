from __future__ import division
import yolfnets as nets
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from yolfnets.references.yolo_utils import get_v2_boxes, v2_loss, v2_inputs
from yolfnets.preprocess import darknet_preprocess as preprocess


def conv(inputs, filters, kernel, strides=1, scope=''):
  x = tf.keras.layers.Conv2D(filters,kernel,strides=strides,activation= tf.keras.activations.relu,padding='same',use_bias=True,name=scope+'/Conv2D',kernel_initializer=tf.keras.initializers.he_normal(seed=1))(inputs)
  x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=scope+'/bnconv2D')(x)
  x = tf.keras.layers.ReLU(max_value=6)(x)
  return x

def sconv(inputs, filters,kernel,strides=1, depth_multiplier=1, scope=''):
  if(filters==None):
    return tf.keras.layers.DepthwiseConv2D(kernel,strides=strides,depth_multiplier=depth_multiplier,padding='same', use_bias=False,name=scope+'/dwconv',kernel_initializer=tf.keras.initializers.he_normal(seed=1))(inputs)
  else:
    return tf.keras.layers.SeparableConv2D(filters,kernel,strides=strides,depth_multiplier=depth_multiplier,padding='same',activation= tf.keras.activations.relu,use_bias=True,name=scope+'/sconv',kernel_initializer=tf.keras.initializers.he_normal(seed=1))(inputs)

def block(inputs, filters, kernel, scope):
  with tf.name_scope(scope): 
    x = sconv(inputs,None,kernel,strides=1,depth_multiplier=1,scope=scope)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=scope+'/bndwconv2D')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = conv(x,filters,1,strides=1,scope=scope)
    return x
    
def meta(dataset_name='voc'):
  if dataset_name=='voc':
    bases = {}
    labels_voc={1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',9:'chair',10:'cow',11:'diningtable',12:'dog',13:'horse',14:'motorbike',15:'person',16:'pottedplant',17:'sheep',18:'sofa',19:'train',20:'tvmonitor'}
    bases['anchors'] =  [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                      8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

    bases.update({'num': 5})
    bases.update({'classes':20, 'labels': labels_voc})
  
  return bases

def tinyYolf(x, is_training, classes, scope='TinyYolf', reuse=None):
  with tf.name_scope(scope):
    x = conv(x, 16, 3, scope='conv1') 
    x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
    x = block(x, 32, 3, scope='conv2') 
    x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
    x = block(x, 64, 3, scope='conv3') 
    x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
    x = block(x, 128, 3, scope='conv4') 
    x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
    x = block(x, 256, 3, scope='conv5') 
    x = tf.keras.layers.MaxPool2D(2,strides=2)(x)
    x = block(x, 512, 3, scope='conv6') 


    x = block(x, 1024, 3, scope='conv7') 
    x = block(x, 1024, 3, scope='conv8')
    x = tf.keras.layers.Conv2D((classes+ 5) * 5, 1, kernel_regularizer=tf.keras.regularizers.l2(5e-4), padding='same', name='genYOLOv2/linear/conv')(x)
    x.aliases = []
    return x


def model(inputs, is_training=True, lmbda=5e-4, dropout_rate=0): 
  metas=meta()
  N_classes=metas['classes']

  print(inputs.shape)
  x = tinyYolf(inputs, is_training=is_training, classes =N_classes , scope='TinyYolf', reuse=None)

  def get_boxes(*args, **kwargs):
  	return get_v2_boxes(metas, *args, **kwargs)
  x.get_boxes = get_boxes
  x.inputs = [inputs]
  x.inputs += v2_inputs(x.shape[1:3], metas['num'], N_classes, x.dtype)
  if isinstance(is_training, tf.Tensor):
      x.inputs.append(is_training)
  x.loss = v2_loss(x, metas['anchors'], N_classes)
  def preprocess_(*args, **kwargs):
  	return preprocess(target_size=(416,416), *args, **kwargs)
  x.preprocess=preprocess_
  return x
