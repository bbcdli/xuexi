# change logs are located in tensor_train.py

import tensorflow as tf
import Image

import cv2
import numpy as np
import keras
#from keras.models import Model
import sys

from keras.backend import set_image_dim_ordering
from keras.models import load_model
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import seg_net_arch as u_a
#from background_learning_s import dice_coef_loss
import time
#tf.python.control_flow_ops = tf #hy:for remote

#KERAS_BACKEND=tensorflow python -c "from keras import backend"
#Using TensorFlow backend.

import Image
import ImageFilter
from functools import wraps
from random import randint
import time
import datetime
import os
import sys

import tensorflow as tf
import cv2
import numpy as np
import PIL

import tflearn as tflearn
from sklearn import datasets
from scipy import ndimage
import math
import operator
import imutils
from PIL import Image #hy: create video with images

import settings #hy: collection of global variables
import prep_image
import tools


#https://keras.io/getting-started/functional-api-guide/
RETRAIN        = 0
CONTINUE_TRAIN = 1

train_1        = 1
train_6c       = 0



ImageType = '.jpg'
#################################################################
# Set seg model
if CONTINUE_TRAIN == 1:
  model_name = 'weights99Jan11_0.04' + '.hdf5'
  set_image_dim_ordering(dim_ordering='th')
#################################################################

def import_data_unet_6c(data_path, file_img, file_mask, h, w, maxNum, do_Flipping=False):
  d  = 3 #d >0: 3-channel, =0: 1-channel, <0:no change
  ch = 3 #1
  print 'load data', data_path, file_img, file_mask, h, w, maxNum, do_Flipping
  images = np.zeros((maxNum*4 , h, w, ch))
  masks  = np.zeros((maxNum*4 , h, w, ch))

  data_counter = 0
  for i in range(1,maxNum+1):
    fmask = data_path + file_mask%i #
    #print i, 'of ',maxNum,'join path and current mask file name:',fmask

    fimg = data_path + file_img%i
    print '\n',i, 'of ',maxNum,'join path and img file name:',fimg

    mask = cv2.imread(fmask, d) # d >0: 3-channel, =0: 1-channel, <0:no change

    img  = cv2.imread(fimg, d)


    if mask is None or img is None:
      continue

    mask = cv2.resize(mask, (h, w))
    img  = cv2.resize(img, (h, w))

    data_counter += 1
    #debug
    #cv2.imshow("img_window",img)
    #cv2.waitKey(100)

    mask = mask.reshape(h, w,d)
    img  = np.float32(img.reshape(h, w,d))
    #debug
    #print '1-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img), np.std(img))

    mask = mask / 255.0
    img  = img / 255.0
    #debug
    #print '2-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img),np.std(img))


    if do_Flipping:
      for fl in range(-1, 2):
        flipped_img = cv2.flip(img, fl)
        flipped_mask = cv2.flip(mask, fl)

        images[data_counter, :, :, :] = flipped_img
        masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
        data_counter += 1

    images[i,:,:,:] = img
    masks[i,:,:,:] = np.float32(mask > 0)

    if i % 100 == 0:
      print 'i=',i
  print 'total', data_counter, 'images and', data_counter, 'masks are loaded'
  #return images, masks
  return images[0:data_counter, :, :, :], masks[0:data_counter, :, :, :]

def import_data_unet_2c(data_path, file_img, file_mask, h, w, maxNum, do_Flipping=False):
  d  = 0 #d >0: 3-channel, =0: 1-channel, <0:no change
  ch = 1 #1
  print 'load data', data_path, file_img, file_mask, h, w, maxNum, do_Flipping
  images = np.zeros((maxNum*4 , ch, h, w))
  masks  = np.zeros((maxNum*4 , ch, h, w))

  data_counter = 0
  for i in range(1,maxNum+1):
    fmask = data_path + file_mask%i #
    #print i, 'of ',maxNum,'join path and current mask file name:',fmask

    fimg = data_path + file_img%i
    print '\n',i, 'of ',maxNum,'join path and img file name:',fimg

    mask = cv2.imread(fmask, d) # d >0: 3-channel, =0: 1-channel, <0:no change

    img  = cv2.imread(fimg, d)


    if mask is None or img is None:
      continue

    mask = cv2.resize(mask, (h, w))
    img  = cv2.resize(img, (h, w))

    data_counter += 1
    #debug
    #cv2.imshow("img_window",img)
    #cv2.waitKey(100)

    mask = mask.reshape(h, w)
    img  = np.float32(img.reshape(h, w))
    #debug
    #print '1-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img), np.std(img))

    mask = mask / 255.0
    img  = img / 255.0
    #debug
    #print '2-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img),np.std(img))


    if do_Flipping:
      for fl in range(-1, 2):
        flipped_img = cv2.flip(img, fl)
        flipped_mask = cv2.flip(mask, fl)

        images[data_counter, :, :, :] = flipped_img
        masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
        data_counter += 1

    images[i,:,:,:] = img
    masks[i,:,:,:] = np.float32(mask > 0)

    if i % 100 == 0:
      print 'i=',i
  print 'total', data_counter, 'images and', data_counter, 'masks are loaded'
  #return images, masks
  return images[0:data_counter, :, :, :], masks[0:data_counter, :, :, :]

def add_colorOverlay(img_grayscale, mask):
  colorOverlay = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB)
  colorOverlay[:, :, 2] = mask
  return colorOverlay

# Create model
def conv2d(img, w, b, k):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'), b))

def max_pool(img, k):
  return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def train_2c(h, w): #input 320x320
  print 'load data'
  images, mask = import_data_unet_2c("../Data/data_3_unet/resized/cad2/", "cad_%03d.jpg", "cad_m_%03d.jpg", h, w, 68,do_Flipping=True)

  '''
  #debug
  for i in xrange(len(images)):# 0,10
    mask   = mask.reshape(h, w)
    images = images.reshape((h, w))
    #cv2.imwrite("input_%03d.jpg" % i, np.uint8(mask) * 255)
    #cv2.imwrite("pic_%03d.jpg" % i, np.uint8(images) * 255)
  '''

  mean = np.mean(images)
  images = images - mean

  stdev = np.std(images)
  images = images / stdev

  print 'mean', mean #  0.506073812469

  print 'stdev', stdev  #0.283976600444

  #images = images.transpose((None, 1, h, w))
  print 'set checkpoint'

  save_params = keras.callbacks.ModelCheckpoint('../testbench/bg/weights' + '{epoch:02d}.hdf5', monitor='val_loss', verbose=2,
                                            save_best_only=False, save_weights_only=False, mode='auto')
  epochs = 1200 #1200
  learning_rate = 0.0002
  decay_rate = learning_rate / epochs
  momentum = 0.99
  sgd = u_a.SGD(lr=learning_rate, momentum=momentum) #hy:decay_rate

  print 'get model setup'
  if RETRAIN == 1:
    model = u_a.unet_arch_2c(h,w)
  if CONTINUE_TRAIN == 1:
    model = load_model("../testbench/bg/" + model_name)

  print 'compile'
  model.compile(loss='binary_crossentropy', optimizer=sgd)
  print 'fit'

  #images.reshape((None,1,h,w))
  #fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None,
  #    shuffle=True, class_weight=None, sample_weight=None)
  #input_X = images.transpose((None,1,h,w))

  model.fit(images, mask, batch_size=1, nb_epoch=epochs, callbacks=[save_params], shuffle=True)

  print 'save'
  #model.save("model2c.h5")
  model.save("../testbench/bg/model2c.h5")


def train_seg_classifier(h, w): #input 320x320
  print 'load data'
  images, mask = import_data_unet_2c("../Data/data_3_unet/resized/hinten/", "cad_%03d.jpg", "cad_m_%03d.jpg", h, w, 65,do_Flipping=True)

  '''
  #debug
  for i in xrange(len(images)):# 0,10
    mask   = mask.reshape(h, w)
    images = images.reshape((h, w))
    #cv2.imwrite("input_%03d.jpg" % i, np.uint8(mask) * 255)
    #cv2.imwrite("pic_%03d.jpg" % i, np.uint8(images) * 255)
  '''

  mean = np.mean(images)
  images = images - mean

  stdev = np.std(images)
  images = images / stdev

  print 'mean', mean #  0.506073812469

  print 'stdev', stdev  #0.283976600444

  #images = images.transpose((None, 1, h, w))
  print 'set checkpoint'

  save_params = keras.callbacks.ModelCheckpoint('../testbench/bg/weights' + '{epoch:02d}.hdf5', monitor='val_loss', verbose=2,
                                            save_best_only=False, save_weights_only=False, mode='auto')
  epochs = 20 #1200
  learning_rate = 0.0002
  decay_rate = learning_rate / epochs
  momentum = 0.99
  sgd = u_a.SGD(lr=learning_rate, momentum=momentum) #hy:decay_rate


  ##############################################################
  with tf.Session() as sess:
    saver = tf.train.Saver()
    if CONTINUE_TRAIN:
    #if RETRAIN:
      # Initializing the variables
      init = tf.initialize_all_variables()  # hy: try
      sess.run(init)
      # Creating a saver for the model
    if CONTINUE_TRAIN == False:  # set model path
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print "Continue to train with ", ckpt.model_checkpoint_path
      else:
        print 'not found model'

    ##############################################################
    print 'get model setup'
    if RETRAIN == 1:
      model = u_a.unet_arch_2c(h,w)
    if CONTINUE_TRAIN == 1:
      model = load_model("../testbench/bg/" + model_name)
  
    print 'compile'
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    print 'fit'
  
    #images.reshape((None,1,h,w))
    #fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None,
    #    shuffle=True, class_weight=None, sample_weight=None)
    #input_X = images.transpose((None,1,h,w))
  
    model.fit(images, mask, batch_size=1, nb_epoch=epochs, callbacks=[save_params], shuffle=True)
  
    #print 'save'
    model.save("../testbench/bg/model1.h5")
    
    labels = []
    images_list = []
    for i in range(0, images.shape[0]):
      result = model.predict(images[i, :, :, :].reshape(1, 1, h, w), batch_size=1)  # old
      
      res = result[0, 0, :, :].reshape((h, w)) * 255
      image_res = np.uint8(res)
      
      if i < 6:
        label = 0
      if i > 6 and i < 9:
        label = 5
      else:
        label = 1
        
      labels.append(label)
      images_list.append(image_res)

      carimages = np.asarray(images_list)
      cartargets = np.asarray(labels)

      digits = datasets.load_digits(n_class=n_classes)
      
      ###########################  CLASSIFIER begin ###################################################
      dropout = [0.3, 0.3, 0.5, 0.5]  # 3,4,5,5
      dropout_1s = [1] * len(dropout)
      n_hidden = 360  # 162*6 # 128
      Graph_3conv = 1
      optimizer_type = 'GD'
      learning_rate = 0.04043  # 0.03549 #0.04049 #0.03049 #0.015 #0.07297 #0.09568# TODO 0.05  0.005 better, 0.001 good \0.02, 0.13799 to 0.14 good for 6 classes,
      if Graph_3conv == 1:
        arch_str = '3conv'
    
      save_all_model = 1
    
      act_min = 0.80
      act_max = 0.93
      add_data = 0  # initial
      area_step_size_webcam = 20  # 479 #200
      set_STOP = False
      stop_loss = 7000.8  # 1.118
      stop_train_loss_increase_rate = 70000.08  # 1.01
      stop_acc_diff = 5  # 3
      stop_acc = 1  # 0.7
    
      last_best_train_acc = 0
      last_best_test_acc = 0
      last_loss = 100

      tensorboard_path = '../Tensorboard_data/sum107/' + str(datetime.now()) + '/'

      model_path_str = 'model_' + optimizer_type + str(n_hidden) + '_h' + \
                       str(settings.h_resize) + '_w' + str(settings.w_resize) \
                       + '_c' + str(6)  # hy include specs of model

      tensor_model_sum_path = '../tensor_model_sum/'

      settings.set_global()
      start_time = time.time()

      current_step = 1

      ######################
      ######################
      # SGD
      lr_decay = 0.01
      decay_step = 100
    
      ###################################################################
      # General input for tensorflow
      # hy: Graph input, same placeholders for various architectures
      tensor_h = 320
      tensor_w = 320
      x = tf.placeholder(tf.float32, [None, tensor_h * tensor_w, 1], name="x")
      y = tf.placeholder(tf.float32, [None, 6], name="y")
      # keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout (keep probability)
      keep_prob = tf.placeholder(tf.float32, len(dropout), name="keep_prob")
      TrainingProp = 0.7
      n_classes = 6

      training_iters = 30000  # 1500  12500,
      if CONTINUE_TRAIN:
        training_iters = current_step + training_iters
    
      ################################################ Graph 3conv begin
      if Graph_3conv == 1:
      
        # tf Graph input
        filter_size_1 = 11
        filter_size_2 = 5
        filter_size_3 = 3
      
        SEED = 8  # hy:  8, 16, 64, number of filters, feature map size: input(42) - filter_size_1 + 1 = 38
        conv2_out = 16  # hy: 16, 32, 64 outputs  of final conv layer, feature map size: input(21) - filter_size_2 + 1 = 19
        conv3_out = 32  # hy: 16, 32, 64 outputs  of final conv layer, feature map size: input(21) - filter_size_2 + 1 = 19
      
        def conv_net(_X, _weights, _biases, _dropout):
          # - INPUT Layer
          # Reshape input picture
        
          _X = tf.reshape(_X,
                          shape=[-1, tensor_h, tensor_w, 1])  # hy: use updated proper values for shape
          print '\nArchitecture\ninput tensor', _X.get_shape()
          # _X = tf.reshape(_X, shape=[-1, 32, 32, 3])  # TODO num channnels change
        
          # a = np.array(_X[0])
          # print(a.shape)
          # Image._show(Image.fromarray(a, 'RGB'))
        
          ################################
          # - Convolution Layer 1
          k = 4
          conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'], k)  # 4
          print 'conv1 ( f=', filter_size_1, 'k=', k, ')', conv1.get_shape()
        
          # Max Pooling (down-sampling)
          k = 2
          conv1 = max_pool(conv1, k)  # TODO return it to K=2
          print 'conv1 max pooling ( k=', k, ')', conv1.get_shape()
          # Apply Dropout
          conv1 = tf.nn.dropout(conv1, _dropout[0])  # TODO comment it later
          print '- dropout ( keep rate', dropout[0], ')', conv1.get_shape()
        
          ################################
          # - Convolution Layer 2
          k = 1
          conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'], k)
          print '\nconv2 ( f=', filter_size_2, 'k=', k, ')', conv2.get_shape()
          # # Max Pooling (down-sampling)
          k = 2
          conv2 = max_pool(conv2, k)
          print 'conv2 - max pooling (k=', k, ')', conv2.get_shape()
          # # Apply Dropout
          conv2 = tf.nn.dropout(conv2, _dropout[1])  # TODO comment it later!
          print '- dropout ( keep rate', dropout[1], ')', conv2.get_shape()
        
          ################################
          # - Convolution Layer 3
          k = 1
          conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'], k)
          print '\nconv3 ( f=', filter_size_3, 'k=', k, ')', conv3.get_shape()
          k = 2
          conv3 = max_pool(conv3, k)
          print 'conv3 - max pooling ( k=', k, ')', conv3.get_shape()
          conv3 = tf.nn.dropout(conv3, _dropout[2])
          print '- dropout ( keep rate', dropout[2], ')', conv3.get_shape()
        
          # Fully connected layer
          dense1 = tf.reshape(conv3,
                              [-1,
                               _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
          print '\ndensel reshape:', dense1.get_shape(), 'n_hidden', n_hidden
          dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))  # Relu activation
          print 'densel - relu:', dense1.get_shape()
        
          dense1 = tf.nn.dropout(dense1, _dropout[3])  # Apply Dropout
          print '- dropout ( keep rate', dropout[3], ')', dense1.get_shape()
        
          # Output, class prediction
          out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
          print 'out:', out.get_shape()
          return out
        
        
        
          # Store layers weight & bias  #Graph_3conv
      
        weights = {
          'wc1': tf.Variable(tf.random_normal([filter_size_1, filter_size_1, 1, SEED], stddev=0.1, seed=SEED), name="wc1"),
          # 5x5 conv, 1 input, 8 outputs
          'wc2': tf.Variable(tf.random_normal([filter_size_2, filter_size_2, SEED, conv2_out], stddev=0.1, seed=SEED),
                             name="wc2"),  # 5x5 conv, 8 inputs, 16 outputs
          'wc3': tf.Variable(tf.random_normal([filter_size_3, filter_size_3, conv2_out, conv3_out], stddev=0.1, seed=SEED),
                             name="wc3"),  # 5x5 conv, 8 inputs, 16 outputs
          # 'wc4': tf.Variable(tf.random_normal([filter_size_4, filter_size_4, conv3_out, conv4_out], stddev=0.1, seed=SEED), name="wc4"),   # 5x5 conv, 8 inputs, 16 outputs
        
          # 'wd1': tf.Variable(tf.random_normal([16 * 24 / 2 * 42 / 2, n_hidden], stddev=0.1, seed=SEED)),  # fully connected, 8*8*64 inputs, 1024 outputs
          # 'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024], stddev=0.1)),  # fully connected, 8*8*64 inputs, 1024 outputs
          'wd1': tf.Variable(tf.random_normal([6 * 6 * conv3_out, n_hidden], stddev=0.1, seed=SEED), name="wd1"),
          # hy: fully connected, 8*8*64 inputs, 1024 outputs
          'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1, seed=SEED), name="w_out")
          # 1024 inputs, 10 outputs (class prediction)
        }
      
        biases = {
          'bc1': tf.Variable(tf.random_normal([SEED]), name="bc1"),
          'bc2': tf.Variable(tf.random_normal([conv2_out]), name="bc2"),  # hy: use variable, instead fixed number
          'bc3': tf.Variable(tf.random_normal([conv3_out]), name="bc3"),  # hy: use variable, instead fixed number
          'bd1': tf.Variable(tf.random_normal([n_hidden]), name="bd1"),
          'out': tf.Variable(tf.random_normal([n_classes]), name="b_out")  # hy:
        }
      
        # hy: try with zero mean
        # tf.image.per_image_whitening(x)
        # this operation computes (x-mean)/adjusted_stddev
      
      
        pred = conv_net(x, weights, biases, dropout)
        # val2_pred = conv_net(x, weights, biases, dropout_1s)
        # pred = conv_net(x, weights, biases, keep_prob)
      
        pred = tf.add(pred, 0, name="pred")
      
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y), name="cost")
        # val2_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
      
        ################ 3conv optimizer
        if optimizer_type == 'GD':
          # learning_rate = tf.train.exponential_decay(learning_rate, step,100000, 0.96, staircase=True)
          # hy: GradientDescentOptimizer
          print '\noptimizer', optimizer_type, '\tlearning_rate', learning_rate
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
      
          print '\noptimizer', optimizer_type, '\tlearning_rate', learning_rate, 'lr_decay', lr_decay, 'decay_step', decay_step
      
        amaxpred = tf.argmax(pred, 1)  # Just to check the bug
        amaxy = tf.argmax(y, 1)  # Just to check for the debug
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
        # val2_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Build the summary operation based on the TF collection of Summaries.
      
        # Adding variables to be visualized
        # hy:add diagrams
        summary = tf.scalar_summary('Accuracy', accuracy)
        tf.scalar_summary('Loss', cost)
      
        ##################################################################################################
        # Tensor VIEW
    
        # _X = np.array(_X[0])
        # tensor_im = cv2.imread('../Data/data_1/hinten/hinten_ww1_rz235_1_ex1_35.jpg')
        # tensor_im = cv2.cvtColor(tensor_im, cv2.COLOR_BGR2GRAY)
        # tensor_im = imutils.resize(tensor_im, width=settings.w_resize, height=settings.h_resize)  # w=146, h=121
        # tensor_im = np.asarray(tensor_im, np.float32)
      
      
        # print(a.shape)
        # Image._show(Image.fromarray(a, 'RGB'))
      
        # tf.image_summary('Images Original',tf.reshape(x, shape=[-1, 24, 42, 1]),max_images=4)
        tf.image_summary('Original', tf.reshape(x, shape=[-1, tensor_h, tensor_w, 1]),
                         max_images=1)  # hy:images_view
      
        # images after conv1 before max pool
        # _X = tf.reshape(x, shape=[-1, 24, 42, 1])
        _X = tf.reshape(x, shape=[-1, tensor_h, tensor_w, 1])  # hy for display
      
        # hy: conv1 view
        # conv1 = tf.placeholder(tf.float32, name="conv1") #hy added
        conv1 = conv2d(_X, weights['wc1'], biases['bc1'], 4)
        conv1 = tf.add(conv1, 0, name="conv1")
        print 'for conv1 view', conv1.get_shape()
      
        conv_view_size = 46
        tf.image_summary('1.Conv', tf.reshape(conv1, shape=[-1, conv_view_size, conv_view_size, 1]), max_images=SEED)  # hy
      
        # hy: conv2 view
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 1)
        conv2 = tf.add(conv2, 0, name="conv2")
        print 'for conv2 view', conv2.get_shape()
        # tf.image_summary('Output of Second Convolution',tf.reshape(conv2, shape=[-1, 24, 42, 1]), max_images=16)
        tf.image_summary('2.Conv', tf.reshape(conv2, shape=[-1, conv_view_size, conv_view_size, 1]),
                         max_images=conv2_out)  # hy
      
        # hy: conv3 view
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 1)
        conv3 = tf.add(conv3, 0, name="conv3")
        print 'for conv3 view', conv3.get_shape()
        tf.image_summary('3.Conv', tf.reshape(conv3, shape=[-1, conv_view_size, conv_view_size, 1]),
                         max_images=conv3_out)  # hy
      
        tf.histogram_summary('Histogram 1.Conv', weights['wc1'])
        # tf.histogram_summary('Histogram 2.Conv', weights['wc2']) #hy: added
        tf.histogram_summary('Histogram pred', pred)  # hy: added
      
        summary_op = tf.merge_all_summaries()
      
        ################################################ Graph 3conv end
      
        ###########################  CLASSIFIER end   ###################################################
      
      ################################################ Graph 3conv end
      ################################### TRAIN begin #####################################################
      if RETRAIN or CONTINUE_TRAIN:
        try:
          #total_images, digits, carimages, cartargets, f, val2_digits, val2_images, val2_targets, val2_f = tools.import_data(
          #  add_online=False)
          #train_size = int(total_images * TrainingProp)
          train_size = 1
          print 'train size', train_size
          batch_size = 1
          # batch_size = int(train_size / n_classes * 2)# *2
    
    
          print 'batch size', batch_size
          val1_batch_xs, val1_batch_ys = digits.images[train_size + 1:1 - 1], \
                                         digits.target[train_size + 1:1 - 1]
    
          '''
          val2_batch_xs, val2_batch_ys = val2_digits.images[0:len(val2_images) - 1], \
                                         val2_digits.target[0:len(val2_images) - 1]  # hy: use calc size
          '''
        except Exception as e:
          print 'Check if file is created correctedly. Setting an array element with a sequence.'
          print str(e)
      
      with tf.Session() as sess:
        saver = tf.train.Saver()
        if RETRAIN:
          # Initializing the variables
          init = tf.initialize_all_variables() #hy: try
          sess.run(init)
          # Creating a saver for the model
        if CONTINUE_TRAIN: #set model path
          ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
          if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Continue to train with ", ckpt.model_checkpoint_path
          else:
            print 'not found model'
    
        elapsed_time = time.time() - start_time
    
        print 'Total elapsed time3:', "{:.2f}".format(elapsed_time), 's'
    
        #hy: added to display all results in one graph
        train_writer = tf.train.SummaryWriter(tensorboard_path + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(tensorboard_path + '/vali', sess.graph)
        test_writer = tf.train.SummaryWriter(tensorboard_path + '/test', sess.graph)
    
    
        #from datetime import datetime
        #tensorboard_path = '../Tensorboard_data/sum107/'+str(datetime.now())+'/'
        #summary_writer = tf.train.SummaryWriter(tensorboard_path, graph_def=sess.graph_def)
    
        if RETRAIN:
          step = 1
        if CONTINUE_TRAIN:
          step = current_step
    
        # hy register finished class learning
        acc_pre = 0
        # Keep training until reach max iterations
        train_size = 1
        batch_size = 1
        while step < training_iters and not set_STOP:
          for batch_step in xrange(int(train_size / batch_size)):
            batch_xs, batch_ys = digits.images[int(batch_step * batch_size):(batch_step + 1) * batch_size -1], \
                                 digits.target[batch_step * batch_size:(batch_step + 1) * batch_size -1]
            print 'batch',batch_step,', from',int(batch_step*batch_size),'to',(batch_step+1)*batch_size-1
            ## Training  ####################################################################
    
            try:
              #hy: feed cusomized value for dropout in training time
              # Calculate batch accuracy, batch loss
              train_acc,loss = sess.run([accuracy,cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    
              # update learning rate if necessary
              print '\n',train_acc, last_best_train_acc, (train_acc - last_best_train_acc)/train_acc
              if (train_acc - last_best_train_acc)/train_acc < 0: #adapt lr
                print 'adapt new lr -'
                learning_rate = tf.train.exponential_decay(learning_rate, step, 10000, 0.99, staircase=True)
                #params:        tf.train.exponential_decay(learning_rate,global_step,decay_steps,decay_rate,staircase)
                #learning_rate = learning_rate*0.999
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                #print sess.run(optimizer.learning_rate)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
              '''
              if (train_acc - last_best_train_acc)/train_acc > 0 and (train_acc - last_best_train_acc)/train_acc < 0.03:
                print 'adapt new lr +'
                learning_rate = 0.04043*1.001
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
              '''
              sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    
            except Exception as e:
              print '\n[Hint] if error occurs, check data input path, settings label size, \ninput tensor size, input for densel' \
                    'is multiplication of the dimension sizes (HxWxD) of previous layer and view size for conv layers, \notherwise, the input tensor size must be changed'
              print '\n[Hint]',str(e)
    
    
            if step % 10 == 0:
    
              elapsed_time = time.time() - start_time
              print 'Up to now elapsed time:', "{:.2f}".format(elapsed_time/ 60), 'min'
    
    
              print "\nIter " + str(step) +'-'+ str(batch_step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " \
                    + "{:.4f}".format(train_acc)
    
              #summary_str = sess.run(summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
              #summary_writer.add_summary(summary_str, step)
    
              ## Validation  ####################################################################
              val1_acc,val1_output = sess.run([accuracy,pred], feed_dict={x: val1_batch_xs, y: val1_batch_ys, keep_prob: dropout_1s})
    
              #val2_acc,val2_loss,val2_output = sess.run([accuracy,cost,pred], feed_dict={x: val2_batch_xs, y: val2_batch_ys, keep_prob: dropout_1s})
    
              print "Validation accuracy=", "{:.4f}".format(val1_acc) #, ','  "test accuracy=", "{:.4f}".format(val2_acc)
    
              #print("Argmax of y:", targetindex)
              #print "Targets:", batch_ys
              #output = tools.convert_to_confidence(output)
    
              #hy: changed normalized=False to True
              confMat, confMat2 = tools.confusion_matrix(val1_batch_ys, val1_output, normalized=True)
              np.set_printoptions(precision=2)  #hy: set display floating point, not changing real value
              print "conf_Mat2 (val1)"
              print confMat2
              tools.print_label_title()
              print confMat #hy: shape n_classes x n_classes
    
              '''
              #print np.sum(confMat)
              confMat, confMat2 = tools.confusion_matrix(batch_ys, output, normalized=True)
              print "conf_Mat2 (test)"
              print confMat2
              tools.print_label_title()
              print confMat  # hy: shape n_classes x n_classes
              '''
    
              #summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
              #summary_writer.add_summary(summary_str, step)
    
              #hy: added to display all results in one graph
              train_res = sess.run(summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_1s})
              train_writer.add_summary(train_res, step)
              
              '''
              val1_res = sess.run(summary_op, feed_dict={x: val1_batch_xs, y: val1_batch_ys, keep_prob: dropout_1s})
              validation_writer.add_summary(val1_res, step)
    
              val2_res = sess.run(summary_op, feed_dict={x: val2_batch_xs, y: val2_batch_ys, keep_prob: dropout_1s})
              test_writer.add_summary(val2_res, step)
              '''
    
              #Retrain hy: control stop
              max_classes_names = []
              max_classes_pres = []
              test_acc_str = 'n'
              name_ext = ''
              sum_col = 0
              sum_col_min = n_classes
              for n in xrange(n_classes):
                max_of_row = max(confMat[n, :])
                max_of_col = max(confMat[:, n])
                diagnal_pres = confMat[n, n]
                if max_of_row == max_of_col and max_of_row == diagnal_pres and sum_col < n_classes:
                  diagnal_pres = round(diagnal_pres, 2)
                  sum_col = sum(confMat[:, n])
                  print 'sum_col:', sum_col, settings.LABEL_names[n]
                  if sum_col < 1.1 and diagnal_pres > 0.6:
                    sum_col_min = min(sum_col_min, sum_col)
                    max_class = settings.LABEL_short[n]
                    max_classes_names.append(max_class)
                    max_classes_pres.append(diagnal_pres)
                    print 'new max value', diagnal_pres, ', class', settings.LABEL_names[n], 'col_sum', sum_col
    
              num_of_classified_classes = len(max_classes_names)
              # print 'collection:',max_classes_names,',',max_classes_pres, ', num:',num_of_classified_classes, 'name_ext:',name_ext
    
              if save_all_model == 1:
                saver.save(sess, save_path=model_path_str + 'all_' + str(batch_step) + '_'+ str(round(val2_acc,2)), global_step=step)  # hy: added. It saves both all variables and GRAPH
    
              if (num_of_classified_classes > 1) or loss < last_loss or train_acc > last_best_train_acc or val2_acc > last_best_test_acc:
                if loss < last_loss:
                  last_loss = loss
                val2_acc = 0# tmp
                if val2_acc > last_best_test_acc:
                  last_best_acc = val2_acc
                  test_acc_str = str(round(last_best_acc, 2))
                if val2_acc > last_best_test_acc:
                  last_best_train_acc = train_acc
                # Save the model
                if num_of_classified_classes > 0 and sum_col_min < 1.1 and val2_acc > last_best_test_acc-0.001 \
                        and loss < 1 and val2_acc > 0.7 and loss < 0.09 or num_of_classified_classes == n_classes:
                  for p in xrange(num_of_classified_classes):
                    name_ext += '_' + max_classes_names[p] + str(max_classes_pres[p])
                  name_ext += '_' + str(batch_step) + '_'+ test_acc_str
                  print 'save model', name_ext
                  # saver.save(sess, save_path=model_path_str + '_I', global_step=step)  # hy: it only saves variables
                  saver.save(sess, save_path=model_path_str + '_' + arch_str + name_ext, global_step=step)  # hy: added. It saves GRAPH
                  cmd = 'mv ../model*' + arch_str + '* ' + tensor_model_sum_path
                  os.system(cmd)
                  cmd = 'rm ../model*'
                  os.system(cmd)
    
              '''
              if val2_acc > 0.3 and (float(val2_loss / loss) > stop_loss
                  or float(train_acc / val2_acc) > stop_acc_diff) \
                  or float(loss/last_loss) > stop_train_loss_increase_rate:
                if float(val2_loss / loss) > stop_loss:
                  print 'Overfitting: loss gap'
                if float(train_acc / val2_acc) > stop_acc_diff:
                  print 'Training will be terminated because of overfitting.'
                if float(loss/last_loss) > stop_train_loss_increase_rate:
                  print 'Training will be terminated because of increasing loss'
    
                set_STOP = True
    
    
              imgNum = len([name for name in os.listdir(settings.data + settings.LABELS[0]) if
                            os.path.isfile(os.path.join(settings.data + settings.LABELS[0], name))])
    
    
              # if (acc - val2_acc) > 0.1 and imgNum < 3* settings.maxNumSaveFiles: #hy: activate random rotation
              if val2_acc > act_min and val2_acc < act_max and imgNum < 2.3 * settings.maxNumSaveFiles:  # hy: activate random rotation
                # rotation_angle = np.random.rand(0, 180) #hy: not working
                rotation_angle = randint(15, 170)
                noise_level = 0.01 * randint(1, 2)
                if imgNum > 2 * settings.maxNumSaveFiles:
                  prep_image.REMOVE_online_Data(step)
                prep_image.rotateflipImg(rotation_angle, 0, noise_level, step)  # hy: angle,flipX,noise_level,step
                add_data = 1
                # training_size = int(total_images * TrainingProp)
                # batch_xs, batch_ys = digits.images[0:training_size], digits.target[0:training_size]
              '''
    
    
          step += 10  #hy: only display every 10th result in console
    
    
        print "\nOptimization Finished!"
  


def train_6classes(h, w):
  print 'train 6classes'
  images_c1, mask_c1 = import_data_unet_6c("../Data/data_3_unet/resized/hinten/", "hinten_%03d.jpg", "hinten_m_%03d.jpg",
                                        h, w, 6, False)
  images_c2, mask_c2 = import_data_unet_6c("../Data/data_3_unet/resized/vorn/", "vorn_%03d.jpg", "vorn_m_%03d.jpg", h, w,
                                        6, False)

  '''
  images_c3, mask_c3 = import_data_unet_6c("../Data/data_3_unet/resized/hinten2/", "hinten_%03d.jpg", "hinten_m_%03d.jpg",
                                        h, w, 2, False)
  

  #images = np.vstack([images_c1, images_c2])
  images = np.zeros((images_c1.shape[0] + images_c2.shape[0] + images_c3.shape[0], 3, 320, 320))
  mask = np.zeros((mask_c1.shape[0] + mask_c2.shape[0] + mask_c3.shape[0], 3, 320, 320))
  
  for i in range(0, images_c1.shape[0]):
    images[i, 0, :, :] =        images_c1[i, :, :, 0]

  for i in range(images_c1.shape[0], images_c1.shape[0] + images_c2.shape[0]):
    images[i, 0, :, :] = images_c2[i, :, :, 0]

  for i in range(images_c1.shape[0] + images_c2.shape[0], images_c1.shape[0] + images_c2.shape[0] + images_c3.shape[0]):
    images[i, 0, :, :] = images_c3[i, :, :, 0]



  for i in range(0, mask_c1.shape[0]):
    mask[i, 0] =        mask_c1[i, :, :, 0]

  for i in range(mask_c1.shape[0], mask_c1.shape[0] + mask_c2.shape[0]):
    mask[i, 1] = mask_c2[i, :, :, 0]

  for i in range(mask_c1.shape[0] + mask_c2.shape[0], mask_c1.shape[0] + mask_c2.shape[0] + mask_c3.shape[0]):
    mask[i, 2] = mask_c3[i, :, :, 0]
  '''

  
  #'''
  images = np.zeros((images_c1.shape[0] + images_c2.shape[0], 3, 320, 320))
  mask = np.zeros((mask_c1.shape[0] + mask_c2.shape[0], 3, 320, 320))
  
  for i in range(0, images_c1.shape[0]):
    images[i, 0, :, :] =        images_c1[i, :, :, 0]
    images[i, 2, :, :] = np.abs(images_c1[i, :, :, 0] - 1.0)

  for i in range(0, images_c2.shape[0]):
    images[i + images_c1.shape[0], 1, :, :] =        images_c2[i, :, :, 0]
    images[i + images_c1.shape[0], 2, :, :] = np.abs(images_c2[i, :, :, 0] - 1.0)
  
  
  for i in range(0, mask_c1.shape[0]):
    mask[i, 0, :, :] =        mask_c1[i, :, :, 0]
    mask[i, 2, :, :] = np.abs(mask_c1[i, :, :, 0] - 1.0)

  for i in range(0, mask_c2.shape[0]):
    mask[i + mask_c1.shape[0], 1, :, :] =        mask_c2[i, :, :, 0]
    mask[i + mask_c1.shape[0], 2, :, :] = np.abs(mask_c2[i, :, :, 0] - 1.0)
  #'''

  mean = np.mean(images)
  images = images - mean
  
  stdev = np.std(images)
  images = images / stdev
  
  print mean
  print stdev
  
  epochs = 10
  learning_rate = 0.0002
  decay_rate = learning_rate / epochs
  momentum = 0.99
  sgd = u_a.SGD(lr=learning_rate, momentum=momentum) #, decay=decay_rate

  save_params = keras.callbacks.ModelCheckpoint('../testbench/bg/weights_6c' + '{epoch:02d}.hdf5', monitor='val_loss',
                              verbose=2, save_best_only=False, save_weights_only=False, mode='auto')
  
  set_image_dim_ordering(dim_ordering='th')

  if RETRAIN == 1:
    model = u_a.unet_arch_6c(h,w)
    #out_a = model(digit_a)
    #out_b = model(digit_b)
    
  if CONTINUE_TRAIN == 1:
    model = load_model("../testbench/bg/" + model_name)

  # from keras.utils.visualize_util import plot
  # plot(model, to_file='model.jpg', show_shapes=True)
  
  #model.compile(loss="binary_crossentropy", optimizer=sgd)
  model.compile(loss="categorical_crossentropy", optimizer=sgd)
  model.fit(images, mask, batch_size=1, nb_epoch=epochs, callbacks=[save_params], shuffle=True) # , validation_split=0.1)
  model.save("model_bin1.h5")
  #######################################


def train_6classes_t1(h, w):
  images_c1, mask_c1 = import_data_unet_6c("../Data/data_3_unet/resized/vorn/", "vorn_%03d.jpg", "vorn_m_%03d.jpg", h, w,
                                        2, True)
  
  images_c2, mask_c2 = import_data_unet_6c("../Data/data_3_unet/resized/hinten/", "hinten_%03d.jpg", "hinten_m_%03d.jpg",
                                        h, w, 2, True)
  
  images = np.vstack([images_c1, images_c2])
  
  mask = np.zeros((mask_c1.shape[0] + mask_c2.shape[0], h * w, 3))
  for i in range(0, mask_c1.shape[0]):
    mask[i, :, 0] = mask_c1[i, 0, :, :].reshape(h * w)
    mask[i, :, 2] = np.abs(mask_c1[i, 0, :, :].reshape(h * w) - 1.0)
  
  for i in range(0, mask_c2.shape[0]):
    mask[i + mask_c1.shape[0], :, 1] = mask_c2[i, 0, :, :].reshape(h * w)
    mask[i + mask_c1.shape[0], :, 2] = np.abs(mask_c2[i, 0, :, :].reshape(h * w) - 1.0)
  
  mean = np.mean(images)
  images = images - mean
  
  stdev = np.std(images)
  images = images / stdev
  
  print mean
  print stdev
  
  epochs = 10
  learning_rate = 0.01
  decay_rate = learning_rate / epochs
  momentum = 0.99
  sgd = u_a.SGD(lr=learning_rate, decay=decay_rate, momentum=momentum)
  
  model = u_a.unet_arch(h, w)

  # from keras.utils.visualize_util import plot
  # plot(model, to_file='model.jpg', show_shapes=True)
  
  model.compile(loss="categorical_crossentropy", optimizer=sgd)
  model.fit(images, mask, nb_epoch=epochs, batch_size=1)
  model.save("model_bin1.h5")


if RETRAIN == 1 or CONTINUE_TRAIN == 1:
  if train_1 == 1:
    #train_seg_classifier(320,320)
    train_2c(320,320)

  if train_6c == 1:
    train_6classes(320,320)
  print("Training done!")


'''
import os
#prepare color mask
#read imgs
path = '../Data/data_3_unet/resized/vorn/'
dirs_c = os.listdir(path)

dirs = [s for s in dirs_c if 'vorn' in s and ImageType in s]
for item in dirs:
  print 'item',path + item
  im_ori = cv2.imread(path + item)
  print 'shape', im_ori.shape
  #im = cv2.resize(np.uint8(im[i, :, :, :].reshape(h, w) * 255), (480, 360))
  #cv2.imshow("test",im_ori)
  #cv2.waitKey(1000)
  
  im = cv2.cvtColor(im_ori, cv2.COLOR_GRAY2RGB)
  cv2.imshow("test",im)
  #cv2.imwrite('./Data/data_3_unet/resized/vorn/' + os.path.basename(item)[:-5] + '_color' + ImageType , im)
print 'files saved in',path, '\n','../Data/data_3_unet/resized/vorn/' + os.path.basename(item)[:-5] + '_color' + ImageType

'''
