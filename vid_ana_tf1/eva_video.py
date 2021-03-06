#eva_video.py
#eva
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
"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data_v as input_data
#import vmodel as c3d_model
import c3d_model
import numpy as np
import cv2
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
FLAGS = flags.FLAGS
#PROJ_DIR = '/home/hy/Documents/liantf1/'
PROJ_DIR = '/home/hy/Documents/aggr/c3d-tf/'
SAVE_RES_TO_FILE = False
NUM_CLASSES = 4
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder
def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var
def run_test():
  #pre_model_name = PROJ_DIR + "models/c3d_ucf_model-1.model"
  model_name = PROJ_DIR + "good_m/" + 'c3d_ucf_model_0.15_0.1-1.meta'
  #model_name = PROJ_DIR + "good_m/c3d_good_model_0.8_0.8-3.meta"
  #test_list_file = PROJ_DIR + 'list/test_eva.list'
  #test_list_file = PROJ_DIR + 'list/test_eva_s.list'
  #num_test_videos = len(list(open(test_list_file,'r')))
  #print "Number of test videos={}".format(num_test_videos)
  # VIDEO_FILE = '/home/hy/Documents/Own_data/basketball/v_shooting_01/' + 'v_shooting_01_01.avi'
  VIDEO_FILE = '/media/sf_shared/vids/done/2017-09-06_13.57.41.5.cam_55_4.event57F.mp4'   #0
  # VIDEO_FILE = '/media/sf_shared/vids/2017-09-06_08.09.58.9.cam_55_4.event12.mp4'
  #VIDEO_FILE = '/media/sf_shared/vids/done/2017-09-06_13.57.41.1.cam_55_3.event46T.mp4'  # 1
  #VIDEO_FILE = '/media/sf_shared/vids/2017-09-06_13.57.41.0.cam_55_3.event43T.mp4'  # 1
  # VIDEO_FILE = '/media/sf_shared/YouTube_DataSet_Annotated/action_youtube_naudio/basketball/v_shooting_01/v_shooting_01_01.avi'   #2
 
  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  #with tf.variable_scope('var_name') as var_scope:
  use_2_layers = True
  use_3_layers = False
  if use_2_layers:
    with tf.variable_scope('var_name') as var_scope:
      weights = {
          'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 4], 0.04, 0.00),
          'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 4, 12], 0.04, 0.00),
          'wd1': _variable_with_weight_decay('wd1', [75264, 128], 0.04, 0.001),
          'wd2': _variable_with_weight_decay('wd2', [128, NUM_CLASSES], 0.04, 0.002),
          'out': _variable_with_weight_decay('out', [NUM_CLASSES, NUM_CLASSES], 0.04, 0.005)
      }
      biases = {
          'bc1': _variable_with_weight_decay('bc1', [4], 0.04, 0.0),
          'bc2': _variable_with_weight_decay('bc2', [12], 0.04, 0.0),
          'bd1': _variable_with_weight_decay('bd1', [128], 0.04, 0.0),
          'bd2': _variable_with_weight_decay('bd2', [NUM_CLASSES], 0.04, 0.0),
          'bout': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
      }
  if use_3_layers:
    with tf.variable_scope('var_name') as var_scope:
      weights = {
          'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 4], 0.04, 0.00),
          'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 4, 12], 0.04, 0.00),
          'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 12, 40], 0.04, 0.00),
          'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 40, 40], 0.04, 0.00),
          'wd1': _variable_with_weight_decay('wd1', [31360, 128], 0.04, 0.001),
          'wd2': _variable_with_weight_decay('wd2', [128, NUM_CLASSES], 0.04, 0.002),
          'out': _variable_with_weight_decay('out', [NUM_CLASSES, NUM_CLASSES], 0.04, 0.005)
      }
      biases = {
          'bc1': _variable_with_weight_decay('bc1', [4], 0.04, 0.0),
          'bc2': _variable_with_weight_decay('bc2', [12], 0.04, 0.0),
          'bc3a': _variable_with_weight_decay('bc3a', [40], 0.04, 0.0),
          'bc3b': _variable_with_weight_decay('bc3b', [40], 0.04, 0.0),
          'bd1': _variable_with_weight_decay('bd1', [128], 0.04, 0.0),
          'bd2': _variable_with_weight_decay('bd2', [NUM_CLASSES], 0.04, 0.0),
          'bout': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
      }
  if not use_2_layers and not use_3_layers:
    with tf.variable_scope('var_name') as var_scope:
      weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, NUM_CLASSES], 0.04, 0.002),
            'out': _variable_with_weight_decay('out', [NUM_CLASSES, NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [NUM_CLASSES], 0.04, 0.0),
            'bout': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  #for gpu_index in range(0, gpu_num):
  gpu_index = 0
  start = time.time()
  with tf.device('/cpu:%d' % gpu_index):
      #dropouts = [1]*5 #3 layers
      dropouts = [1]*4 #2 layers
      logit = c3d_model.inference_c3d_2l(
          images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :],
          dropouts,
          FLAGS.batch_size,
          weights,
          biases
      )
      #logit = c3d_model.inference_c3d_2l(
      # images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
      # dropouts, FLAGS.batch_size, weights, biases)
      logits.append(logit)

  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  print 'got norm score'
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  #model path_file: PROJ_DIR + "models/c3d_ucf_model-1.meta"
  saver.restore(sess, model_name[:-5])
  print 'restore ok'
  # And then after everything is built, start the loop.
  bufsize = 0
  res_name,end = os.path.splitext(os.path.basename(model_name))
  res_name = 'pred_res' + res_name
  if SAVE_RES_TO_FILE:
    write_file = open(res_name + '.txt', "w+", bufsize)
  print ('time before test loop:', time.time() - start, 's')

  video = cv2.VideoCapture(VIDEO_FILE)
  video.set(1,2)
  if not video.isOpened():
    print 'error when reading video file'
    exit(-1)
  else:
    print 'video loaded:', VIDEO_FILE
  v_frame_i, sub_count, sub_frames = 0, 0, []
  while True:
      ret,frame = video.read()
      if ret:
          fh,fw = frame.shape[0], frame.shape[1]
          #print 'fh,fw:',fh,fw #240, 320
          v_frame_i += 1
      else:
          break
      if v_frame_i % 1 == 0:
        sub_count += 1
        #prepare test clip in form of 16-frames
        sub_frames.append(frame)
        if len(sub_frames) >= 16:
          print '\nclip of 16 frames:',sub_count,' ',
          test_images, test_labels = input_data.read_clip_and_label_video(sub_frames, 16)
          #print 'len of test images',len(test_images),
          sub_frames = []
          inference_before = time.time()
          predict_score = norm_score.eval(
                  session=sess,
                  feed_dict={images_placeholder: test_images}
                  )
          for i in range(0, 1): #
            true_label = test_labels[i],
            top1_predicted_label = np.argmax(predict_score[i])
            # Write results: true label, class prob for true label, predicted label, class prob for predicted label
            #gt, corr_rate, pred_label, confidence

            print 'target:',true_label[0], 'pred:',top1_predicted_label
            if SAVE_RES_TO_FILE:
              write_file.write('{}, {}, {}, {}\n'.format(
                    true_label[0],
                    predict_score[i][true_label],
                    top1_predicted_label,
                    predict_score[i][top1_predicted_label]))
              print ('inference time for 1 clip:%.4f' % (time.time() - inference_before), 's')
      #if v_frame_i > 50:
      #  print 'quit'
      #  break
      ret,frame = video.read()
      if not ret:
        print 'video end,quit'
        break
  if SAVE_RES_TO_FILE:
    write_file.close()
  print("done")
def main(_):
  run_test()
if __name__ == '__main__':
  tf.app.run()



