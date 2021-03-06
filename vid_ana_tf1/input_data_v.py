#input_data_v.py
#small dataset: http://crcv.ucf.edu/data/UCF_YouTube_Action.php
#input_data.py
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
#def get_frames_data(filenames, num_frames_per_clip=16):
def get_frames_data_random(lkdir, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  #print ('len of files',len(filenames),', filenames:',filenames[-10:])
  for dir, subdir, filenames in os.walk(lkdir):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    #filenames = sorted(filenames)
    s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(s_index, s_index + num_frames_per_clip):
      image_name = str(dir) + '/' + str(filenames[i])
      #print 'image:', image_name
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)
  return ret_arr, s_index

#collect images of given path for one clip
def get_frames_data_read_clipline(lkdir, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  #print 'lkdir', lkdir
  im_dirs = sorted([s for s in os.listdir(lkdir)])
  s_index = random.randint(0, len(im_dirs) - num_frames_per_clip)
  if len(im_dirs) < num_frames_per_clip:
    return [], s_index
  for i,im in zip(range(s_index, s_index + num_frames_per_clip),im_dirs):
    image_name = lkdir +'/'+ im
    #print 'image:', image_name
    img = Image.open(image_name)
    img_data = np.array(img)
    ret_arr.append(img_data)
  return ret_arr, s_index

def reduce_mean_stdev(images, print_val=False):
    mean = np.mean(images)
    stdev = np.std(images)
    if print_val:
        print 'mean %d,stdev %d', (mean, stdev)
    images = images - mean
    images_reduced_mean = images / stdev
    return images_reduced_mean

def read_clip_and_label_v2(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112,
                               shuffle=False):
  lines = open(filename, 'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  print 'len lines', len(lines)
  # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if (batch_index >= batch_size):
      next_batch_start = index
      # print 'batch pos > batch_size', batch_index
      break
    # split line by ' '
    line = lines[index].strip('\n').split()
    dir = line[0]
    tmp_label = line[1]
    if not shuffle:
      #print("Loading a video clip from {}...".format(dir))
      pass #hy

    tmp_data, _ = get_frames_data_read_clipline(dir, num_frames_per_clip)
    # print 'got frames data, len data', len(tmp_data)
    img_datas = []
    if (len(tmp_data) != 0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if (img.width > img.height):
          scale = float(crop_size) / float(img.height)
          img = np.array(cv2.resize(np.array(img), (112, 112))).astype(np.float32)
        else:
          scale = float(crop_size) / float(img.width)
          img = np.array(cv2.resize(np.array(img), (112, 112))).astype(np.float32)
          #img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        np_mean = reduce_mean_stdev(img)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dir)
  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))
  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  # print 'next' ,next_batch_start
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

def read_clip_and_label_npmean(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  print 'len lines', len(lines)
  #np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      #print 'batch pos > batch_size', batch_index
      break
    #split line by ' '
    #print 'line index ', index
    line = lines[index].strip('\n').split()
    dir,tail = os.path.split(line[0])
    tmp_label = line[1]
    #print 'clipdir, label:', dir,tmp_label
    if not shuffle:
      #print("Loading a video clip from {}...".format(dir))
      pass #hy
  
    tmp_data, _ = get_frames_data_read_clipline(dir, num_frames_per_clip)
   
    #print 'got frames data, len data', len(tmp_data)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(112, 112))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        np_mean = reduce_mean_stdev(img)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dir)
  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))
  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #print 'next' ,next_batch_start
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len
def read_clip_and_label_video(frames, num_frames_per_clip=16, crop_size=112, shuffle=False):
  '''
  collect num_frames_per_clip=16 frames
  '''
  data,img_datas, label = [],[], []
  #np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
 
  tmp_data = np.array(frames)
  target_label = 0  #
 
  if(len(tmp_data)!=0):
    for im,j in zip(frames,xrange(num_frames_per_clip)):
      img = Image.fromarray(im.astype(np.uint8))
      #print 'img w,h',img.size  #320,240
      #if(img.width<img.height):  #fh,fw=240,320
      if(img.width>img.height):
        scale = float(crop_size)/float(img.height)
        img = np.array( cv2.resize(np.array(img), (112, 112))).astype(np.float32)
        #img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
      else:
        scale = float(crop_size)/float(img.width)
        img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        np_mean = reduce_mean_stdev(img)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean
        #img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
      img_datas.append(img)
      data.append(img_datas)
      label.append(int(target_label))
 
  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #print 'np data shape:',np_arr_data.shape
  return np_arr_data, np_arr_label

def read_clip_and_label_images(images, target_label, num_frames_per_clip=16, crop_size=112):
  '''
  collect num_frames_per_clip=16 frames
  '''
  data, img_datas, label = [], [], []
  # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  tmp_data = np.array(images)
  if (len(tmp_data) != 0):
    for im, j in zip(images, xrange(num_frames_per_clip)):
      img = Image.fromarray(im.astype(np.uint8))
      # print 'img w,h',img.size  #320,240
      # if(img.width<img.height):  #fh,fw=240,320
      if (img.width > img.height):
        scale = float(crop_size) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (112, 112))).astype(np.float32)
        # img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
      else:
        scale = float(crop_size) / float(img.width)
        img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        np_mean = reduce_mean_stdev(img)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean
        # img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
      img_datas.append(img)
      data.append(img_datas)
      label.append(int(target_label))
  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #np_arr_data = np.array(data)
  #np_arr_label = np.array(label)
  # print 'np data shape:',np_arr_data.shape
  return np_arr_data, np_arr_label



