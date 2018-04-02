import cv2
# change logs are located in tensor_train.py
import keras
import keras.backend as kb
from keras.backend import set_image_dim_ordering
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from itertools import *  # for izip
import seg_arch as arch
import random
from PIL import ImageFilter
from random import randint
import time
import datetime
import os
import sys
import numpy as np
import PIL
from PIL import Image
import tools
import tensorflow as tf
# https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
# https://keras.io/getting-started/functional-api-guide/
from keras.preprocessing.image import ImageDataGenerator

do_reduce_mean = True


def load_and_preprocess_data_k(h, w, PROJ_DIR, data_path, rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min,
                               rnd_darkness_max):
  print 'train_path:', data_path
  data_path = [x.strip() for x in data_path.split(',')]
  print 'train_paths:', data_path

  #####################################################################################
  im_path = os.path.join(PROJ_DIR, 'Data/training/', data_path[0], '1_im/')
  m_path = os.path.join(PROJ_DIR, 'Data/training/', data_path[0], '1_m/')

  data_1s = sorted([s for s in os.listdir(im_path) if '_s' not in s])
  m_1s = sorted([s for s in os.listdir(m_path) if '_s' not in s])

  # data_1s = data_1s[0:6]
  # m_1s = m_1s[0:6]

  images, masks = tools.import_data_k_segnet(im_path, m_path, data_1s, m_1s, h, w, len(data_1s),
                                             rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min, rnd_darkness_max,
                                             do_Flipping=True, do_gblur=True, do_darken=True)
  if do_reduce_mean:
    images = tools.reduce_mean_stdev(images)
  #####################################################################################

  add_data_2 = True
  if add_data_2:
    im_path2 = os.path.join(PROJ_DIR, 'Data/training/', data_path[0], '1_im/')  #
    m_path2 = os.path.join(PROJ_DIR, 'Data/training/', data_path[0], '1_m/')  #

    data_2s = sorted([s for s in os.listdir(im_path2) if '_s' in s])
    m_2s = sorted([s for s in os.listdir(m_path2) if '_s' in s])

    len_f = len(data_2s)
    total_files_im, total_files_m, rnd_files_im, rnd_files_m = [], [], [], []
    rnd_indx = random.sample(xrange(len_f - 1), int(0.5 * len_f))  # population,length
    print 'rnd_indx:', rnd_indx
    for i in rnd_indx:
      rnd_files_im.append(data_2s[i])
      rnd_files_m.append(m_2s[i])

    total_files_im = total_files_im + rnd_files_im
    total_files_m = total_files_m + rnd_files_m

    data_2s, m_2s = total_files_im, total_files_m
    images2, mask2 = tools.import_data_k_segnet(im_path2, m_path2, data_2s, m_2s, h, w, len(data_2s),
                                                rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min,
                                                rnd_darkness_max,
                                                do_Flipping=True, do_gblur=True, do_darken=True)
    print 'train_path:', im_path2, ', images2 shape:', images2.shape, ', mask2 shape:', mask2.shape
    if do_reduce_mean:
      images2 = tools.reduce_mean_stdev(images2)
    images = np.concatenate((images, images2), axis=0)
    masks = np.concatenate((masks, mask2), axis=0)

  #####################################################################################

  add_data_3 = True
  if add_data_3:
    im_path3 = os.path.join(PROJ_DIR, 'Data/training/', data_path[1], '2_im/')  #
    m_path3 = os.path.join(PROJ_DIR, 'Data/training/', data_path[1], '2_m/')  #
    data_3s = sorted([s for s in os.listdir(im_path3)])
    m_3s = sorted([s for s in os.listdir(m_path3)])

    images3, mask3 = tools.import_data_k_segnet(im_path3, m_path3, data_3s, m_3s, h, w, len(data_3s),
                                                rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min,
                                                rnd_darkness_max,
                                                do_Flipping=True, do_gblur=True, do_darken=True)
    print 'train_path:', im_path3, ', images3 shape:', images3.shape, ', mask3 shape:', mask3.shape
    if do_reduce_mean:
      images3 = tools.reduce_mean_stdev(images3)
    images = np.concatenate((images, images3), axis=0)
    masks = np.concatenate((masks, mask3), axis=0)

  #####################################################################################
  ####################################################################################

  print 'images shape after mean reduction:', images.shape
  return images, masks


def load_and_preprocess_data_onlinetest_k(h, w, PROJ_DIR, test_data_path, rnd_blurriness_min, rnd_blurriness_max,
                                          rnd_darkness_min, rnd_darkness_max):
  total_files_im, total_files_m, rnd_files_im, rnd_files_m = [], [], [], []
  # all_read_path_im, all_read_path_m = [], []
  folders = ['1/']

  for folder in folders:
    read_path_im = os.path.join(PROJ_DIR, test_data_path, folder, 'im/')
    read_path_m = os.path.join(PROJ_DIR, test_data_path, folder, 'm/')
    files_im = sorted([s for s in os.listdir(read_path_im) if '.' in s])
    files_m = sorted([s for s in os.listdir(read_path_m) if '.' in s])
    print 'len of online test:', len(files_im)

    len_f = len(files_im)
    rnd_indx = random.sample(xrange(len_f - 1), int(0.9 * len_f))  # population,length
    print 'rnd_indx:', rnd_indx
    for i in rnd_indx:
      rnd_files_im.append(files_im[i])
      rnd_files_m.append(files_m[i])

    total_files_im = total_files_im + rnd_files_im
    total_files_m = total_files_m + rnd_files_m

    # for i in xrange(len(files_im)):
    #  all_read_path_im.append(read_path_im)
    #  all_read_path_m.append(read_path_m)

  images_t, masks_t = tools.import_data_k_segnet(read_path_im, read_path_m, total_files_im, total_files_m, h, w,
                                                 len(total_files_im), rnd_blurriness_min, rnd_blurriness_max,
                                                 rnd_darkness_min, rnd_darkness_max,
                                                 do_Flipping=False, do_gblur=False, do_darken=False)
  if do_reduce_mean:
    images_t = tools.reduce_mean_stdev(images_t)
  return images_t, masks_t


def train_2c(PROJ_DIR, train_mode, model_path, model_con_name, data_path, test_data_path, log_LABEL,
             learning_rate, batch_size, MAX_ITERATION, INPUT_SIZE,
             rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min, rnd_darkness_max, dropouts):
  print ('PROJ_DIR,train_mode,model_path,model_con_name,data_path,test_data_path,log_LABEL,\
    learning_rate,batch_size,MAX_ITERATION,INPUT_SIZE,rnd_blurriness_min,rnd_blurriness_max,\
    rnd_darkness_min,rnd_darkness_max,dropouts',
         PROJ_DIR, train_mode, model_path, model_con_name, data_path, test_data_path, log_LABEL,
         learning_rate, batch_size, MAX_ITERATION, INPUT_SIZE,
         rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min, rnd_darkness_max, dropouts)

  model_path_name = os.path.join(model_path, model_con_name)

  if 'con' in train_mode:
    # manual
    print '\nfollow model:', model_path_name, 'learning rate:', learning_rate
    set_image_dim_ordering(dim_ordering='th')

  h, w = INPUT_SIZE, INPUT_SIZE
  if 'con' not in train_mode:
    model_path_name = model_path + '.hdf5'

  print '\ntrain binary classes, load data,learning rate:', learning_rate

  images, masks = load_and_preprocess_data_k(h, w, PROJ_DIR, data_path,
                                             rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min, rnd_darkness_max)
  print 'train len:', len(images)

  images_t, masks_t = load_and_preprocess_data_onlinetest_k(h, w, PROJ_DIR, test_data_path,
                                                            rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min,
                                                            rnd_darkness_max)

  print 'images shape', images.shape

  # images = images.transpose((None, 1, h, w))
  print 'set checkpoint'

  save_params = ModelCheckpoint(filepath=model_path + log_LABEL + '_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val_loss', verbose=2,
                                save_best_only=False, save_weights_only=False, mode='auto')

  keras.callbacks.History()
  epochs = MAX_ITERATION  # 
  learning_rate = learning_rate  # 
  # decay_rate = learning_rate / epochs #not so good
  momentum = 0.99
  sgd = arch.SGD(lr=learning_rate, momentum=momentum)  #
  set_image_dim_ordering(dim_ordering='th')

  print 'load network'
  if train_mode == 'new_train':
    model = arch.segnet_arch_2c(dropouts, h, w)

  if train_mode == 'con_train':
    model = load_model(model_path_name)

  print 'compile'
  # compile option 1
  model.compile(loss='binary_crossentropy', optimizer=sgd)

  model.fit(images, masks, batch_size=batch_size, nb_epoch=epochs, verbose=1, shuffle=True,
            validation_data=(images_t, masks_t), callbacks=[save_params])

  # visulization
  # verbose=1 to switch on printing batch result

  print 'save'
  model.save(model_path + 'model_' + log_LABEL + '.h5')


def main(_):
  PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
  train_mode = sys.argv[1]
  model_path = sys.argv[2]
  model_con_name = sys.argv[3]
  data_path = sys.argv[4]
  test_data_path = sys.argv[5]
  log_LABEL = sys.argv[6]
  learning_rate = float(sys.argv[7])
  batch_size = int(sys.argv[8])
  MAX_ITERATION = int(sys.argv[9])  # int(1e6 + 1)
  INPUT_SIZE = int(sys.argv[10])
  rnd_blurriness_min = int(sys.argv[11])
  rnd_blurriness_max = int(sys.argv[12])
  rnd_darkness_min = int(sys.argv[13])
  rnd_darkness_max = int(sys.argv[14])
  dropouts = sys.argv[15]

  train_2c(PROJ_DIR, train_mode, model_path, model_con_name, data_path, test_data_path, log_LABEL,
           learning_rate, batch_size, MAX_ITERATION, INPUT_SIZE,
           rnd_blurriness_min, rnd_blurriness_max, rnd_darkness_min, rnd_darkness_max, dropouts)
  print("Training done!")


if __name__ == '__main__':
  tf.app.run()

