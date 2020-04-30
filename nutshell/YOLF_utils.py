from __future__ import division
import yolfnets as nets
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from yolfnets.references.yolo_utils import get_v2_boxes, v2_loss, v2_inputs
from yolfnets.preprocess import darknet_preprocess as preprocess

def darkdepthsepconv(inputs, filters, kernel, name, lmbda=5e-4, dropout_rate=0):
  with tf.name_scope(name):
    x = tf.keras.layers.DepthwiseConv2D(kernel, depth_multiplier=1, padding='same', use_bias=False, name=name+'/sconv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=name+'/bnd')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False, name=name+'/conv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=name+'/bns')(x)
    x = tf.nn.bias_add(x, tf.Variable(tf.random_normal([filters])), name= name+'bias_add')
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
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

def model(inputs, is_training=True, lmbda=5e-4, dropout_rate=0): 
  metas=meta()
  N_classes=metas['classes']
  lmbda += 1e-10

  with tf.name_scope('stem'):
    x = stem = nets.MobileNet25(inputs, is_training=True, stem=True,  scope='stem', lmbda=lmbda, dropout_rate=dropout_rate) #bulding the model

  p = x.p

  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv7', lmbda=lmbda, dropout_rate=dropout_rate)
  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv8', lmbda=lmbda, dropout_rate=dropout_rate)

  p = darkdepthsepconv(p, 128, 1, name='genYOLOv2/conv5a', lmbda=lmbda, dropout_rate=dropout_rate)
  p = tf.reshape(p,[-1, 13,13,512], name='flat5a')
  x = tf.concat([p, x], axis=3, name='concat')

  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv9', lmbda=lmbda, dropout_rate=dropout_rate)
  x = tf.keras.layers.Conv2D((N_classes+ 5) * 5, 1, kernel_regularizer=tf.keras.regularizers.l2(lmbda), padding='same', name='genYOLOv2/linear/conv')(x)
  x.aliases = []

  def get_boxes(*args, **kwargs):
  	return get_v2_boxes(metas, *args, **kwargs)
  x.get_boxes = get_boxes
  x.stem = stem
  x.inputs = [inputs]
  x.inputs += v2_inputs(x.shape[1:3], metas['num'], N_classes, x.dtype)
  if isinstance(is_training, tf.Tensor):
      x.inputs.append(is_training)
  x.loss = v2_loss(x, metas['anchors'], N_classes)
  def preprocess_(*args, **kwargs):
  	return preprocess(target_size=(416,416), *args, **kwargs)
  x.preprocess=preprocess_
  return x
