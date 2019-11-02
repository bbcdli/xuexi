# originally by Hamed, 25Apr.2016
# hy:Changes by Haiyan, 21Dec.2016 v0.45
# sudo apt-get install python-h5py
# Added evaluation function for multiple models, their result file names contain calculated mAP.
# Added functionality to set different dropout rate for each layer for 3conv net
# Moved auxiliary functions to a new file tools.py
# Added function to obtain images of estimated receptive fields/active fields
# Added function to save all models and specified names according to training status
# Added graph 3conv, 4conv
# Added real batch training functionality
# Added functionality of feeding a tensor name
# Added function to save tensorflow models with max precision for a class, not overwritten by following data
# Added function do_crop2_parts to get parts in different sizes
# Added function for displaying evaluation results in a worksheet (result_for_table = 0).
# Added similarity.py to analyse similarity between classes, CAD sampls and camera test images
# Created tensor_cnn_evaluate.py. It is used for testing multiple models. Input of each evaluation function includes:
#   session,num_class,img_list,_labels
# Added stop condition to avoid overfitting
# Added function to load two models of different graphs. requirement: install tensorflow version > 0.8, numpy > 1.11.2
# Added display of all defined results for training, validation and test in one graph in tensorboard
# Added optimizer Adam and its parameters
# Added display of test result in RETRAIN
# Added a function to add more training data during a training. This data contains random noise.
# Added display of test result in CONTINUE_TRAIN. Some new variables are created for tensorflow for this purpose.
# Created a function for importing data, import_data(). This is used for displaying test result parallel to validation result.
# Added function to evaluate two models of same graph
# Added adaptive testing - evaluate_image_vague, create_test_slices to get top,bottom, left, right, center parts of a test image
# Added formula for calculating window size when webcam is used, also for rectangular form
# Added functions: random crop, random rotation, set scale, remove small object area
# Added def convert_result for converting sub-class to main-class result.
# Changed tensorboard backup path and added sub-folder to store tensorboard logs so that the logs can be compared easily.
# Changed model name to include specification info of a model.
# Specification information of a model such as number of hidden layers and tensor size must be set as the same when this model is reused later.
# Added functionality of continuing a broken training
# Added distortion tools for automatically generating and moving/removing data
# Added tensorboard log timestamp for comparing different model in live time, changed tensorboard log path
# Added function to do tracking in terms of shift mean #
# Added date time for log
# Training set: CAD samples for all six classes
# Added functionality of saving first convolutional layer feature output in training phase and test phase
# Added function to evaluate model with webcam
# Prepare_list is activated according to action selected for training or test
# Test set: lego positive samples for all six classes
# Added output info: when evaluating with images, proportion of correctly classified is included
# Added sequence configurations for based on training or test which is selected
# Added function to save correctly classified images/frames
# Added function to save misclassified images to folder ../MisClassifed, upper limit can be set
# Added log function, time count for training duration
# Test_Images: stored under ../Test_Images, they are lego positive samples that are not included in training set.
# Added the functionality to evaluate model with images
# Changed prepare_list to a global function to make test run smoothly.
# Changed condition for label, predict
# Changed display precision of matrix outputs to 2
# Added a formula to calculate shape, in settings.py
# Added a formula to set cropped frame to show ROI in demo
# Tested video_crop_tool.py, it does not require strict parameter for width as in this script
# Added global variables for width, height, crop sizes, defined in settings.py
# Changed places to adapt to lego data
#   - All file paths in tensor_cnn_video.py, prepare_list.py, image_distortions.py, test.py
#   - LABELS(=6), which is the number of sub-folders under ../Data
# To see tensorflow output use following command
# $tensorflow --logdir='enter_the_path_of_tensorboard_log'
#####################################################################################################

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
from sklearn import datasets
from scipy import ndimage
import math
import operator
import imutils
from PIL import Image  # hy: create video with images

import settings  # hy: collection of global variables
import prep_image
import tools

# activate global var
settings.set_global()
start_time = time.time()
# http://lvdmaaten.github.io/tsne/ visualization
## Train or Evaluation


############################################################
RETRAIN = True
current_step = 141  # 4311, 4791,1211, 3271, 3491, 21291 72.4 model_60_h18_w18_c8-79302-top

# Network Parameters
#learning_rate = 0.02509  # 0.03049 #0.015 #0.07297 #0.09568# TODO 0.05  0.005 better, 0.001 good \0.02, 0.13799 to 0.14 good for 6 classes,
# #0.13999 (flat) to 0.13999  (gap) for 7 classes, 0.0699965 for 6 classes with fine samples
# 0.0035 for links+rechts 98%;

#n_hidden = 360  # 162*6 # 128
# 300: horizontal 20%
# 360: until 1200 step good, after that test acc remains
# 200: start to increase early, 200, but does not increase lot any more
# 150, 250, 300, 330, 400: until 70 iter 17%

# Select architecture
Graph_2conv = 0
Graph_3conv = 1
Graph_3conv_same_dropout = 0
Graph_4conv = 0

if Graph_2conv == 1:
 arch_str = '2conv'
if Graph_3conv == 1 or Graph_3conv_same_dropout == 1:
 arch_str = '3conv'
 #if Graph_3conv == 1:
  #dropout = [0.25, 0.25, 0.25, 0.25]  # 3,4,5,5
  #dropout = [0.25] #3,4,5,5
  #dropout_1s = [1] * len(dropout)
 # dropout = 0.5 # Dropout, probability to keep units
if Graph_4conv == 1:
 arch_str = '4conv'

save_all_models = 1

act_min = 0.80
act_max = 0.93
add_data = 0  # initial
area_step_size_webcam = 20  # 479 #200
#optimizer_type = 'GD'  # 'adam' #GD-'gradient.descent'
set_STOP = False
stop_loss = 7000.8  # 1.118
stop_train_loss_increase_rate = 70000.08  # 1.01
stop_acc_diff = 5  # 3
stop_acc = 1  # 0.7

last_best_acc = 0
last_best_test_acc = 0
last_loss = 100

CONTINUE_TRAIN = True
GENERATE_FILELIST = 1

log_on = True
DEBUG = 1
TrainingProp = 0.70

###########################################################################################################
# the rest functions are also separately located in *evaluation* file, they will be updated only sometimes.
###########################################################################################################
TEST_with_Webcam = False  # hy True - test with webcam
video_label = 0  # hy: initialize/default 0:hinten 1:links 2:'oben/', 3:'rechts/', '4: unten/', 5 'vorn/

TEST_with_Images = False  # hy True - test with images
TEST_with_Video = False  # hy True - test with video
video_window_scale = 2

TEST_CONV_OUTPUT = False
result_for_table = 0

SAVE_Misclassified = 0
SAVE_CorrectClassified = 0

# Input data
# n_input = 42 * 42  # Cifar data input (img shape: 32*32)
n_input = settings.h_resize * settings.w_resize  # hy
n_classes = len(settings.LABELS)  # hy: adapt to lego composed of 6 classes. Cifar10 total classes (0-9 digits)
# Noise level
noise_level = 0

# Data
LABEL_LIST = settings.data_label_file
LABEL_PATH = settings.data_label_path

LABEL_LIST_TEST = settings.test_label_file
LABEL_PATH_TEST = settings.test_label_path

LABELS = settings.LABELS  # hy
LABEL_names = settings.LABEL_names  # hy

# Active fields test for visualization
do_active_fields_test = 0
if do_active_fields_test == 1:
 print 'To get active fields analysis you must set read_images to sorted read'
 LABEL_PATH_TEST = "../Test_Images/test_active_fields/*/*"  #
 LABEL_LIST_TEST = settings.test_label_file_a
 activation_test_img_name = '../Test_Images/hinten_ori1_rz400.jpg'

# auto-switches  #########################
if RETRAIN or TEST_with_Images or TEST_with_Webcam or TEST_with_Video:
 CONTINUE_TRAIN = False

if RETRAIN or CONTINUE_TRAIN:
 TEST_with_Images = False
 TEST_with_Webcam = False
 TEST_with_Video = False
 do_active_fields_test = 0
#########################################


# hy:add timestamp to tensor log files
from datetime import datetime

tensorboard_path = '../Tensorboard_data/sum107/' + str(datetime.now()) + '/'

tensor_model_sum_path = '../tensor_model_sum/'
#classifier_model = "../logs/" + "model_GD360_h184_w184_c6all_10_0.71-191_reprod_dropoutList_part2.meta"

if GENERATE_FILELIST == 1:
 # image_distortions.rotateflipImg()
 if RETRAIN or CONTINUE_TRAIN:
  tools.prepare_list(LABEL_LIST, LABEL_PATH)  # hy: avoid wrong list error #hy trial
  tools.read_images(LABEL_LIST)  # hy: get usable input size for w,h
  
  tools.prepare_list(LABEL_LIST_TEST, LABEL_PATH_TEST)  # hy: avoid wrong list error #hy trial
  tools.read_images(LABEL_LIST_TEST)  # hy: get usable input size for w,h
 
 else:
  print 'Creating test file list', LABEL_LIST_TEST, 'path', LABEL_PATH_TEST
  tools.prepare_list(LABEL_LIST_TEST, LABEL_PATH_TEST)  # hy: avoid wrong list error #hy trial
  if do_active_fields_test == 1:
   tools.read_images(LABEL_LIST_TEST, random_read=False)  # hy: get usable input size for w,h
  else:
   tools.read_images(LABEL_LIST_TEST, random_read=True)  # hy: get usable input size for w,h

else:
 if TEST_with_Images or TEST_with_Video:
  tools.read_images(LABEL_LIST_TEST)
 # tools.read_image_output_slices(LABEL_LIST_TEST) #hy: get slices for activation analysis
 else:
  tools.read_images(LABEL_LIST)

training_iters = 300000#151  # 30000 # 1500  12500,
if CONTINUE_TRAIN:
 training_iters = current_step + 90000 #90000010

display_step = 1  # a factor, will be multiplied by 10

print 'classes:', settings.LABELS


def track_roi(VIDEO_FILE):
 video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
 # cv2.waitKey(10)
 
 
 video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
 # hy: propID=1 means 0-based index of the frame to be decoded/captured next
 
 if not video.isOpened():
  print "cannot find or open video file"
  exit(-1)
 
 # Read the first frame of the video
 ret, frame = video.read()
 
 # Set the ROI (Region of Interest). Actually, this is a
 # rectangle of the building that we're tracking
 c, r, w, h = 900, 650, 400, 400
 track_window = (c, r, w, h)
 
 # Create mask and normalized histogram
 roi = frame[r:r + h, c:c + w]
 hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
 mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
 
 roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
 
 cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
 
 term_cond = (
  cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)  # hy: TERM_CRITERIA_EPS - terminate iteration condition
 
 while True:
  ret, frame = video.read()
  if ret:
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
   ret, track_window = cv2.meanShift(dst, track_window, term_cond)
   
   x, y, w, h = track_window
   
   # hy: draw rectangle as tracked window area
   cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
   cv2.putText(frame, 'Tracked', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
               1, (255, 255, 255), 2, cv2.CV_AA)
   
   cv2.imshow('Tracking', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  else:
   print 'no frame received'
   break
 
 return [track_window]


def EVALUATE_IMAGES_sort_activation(sess):
 # Testing
 carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST)
 # carimages, cartargets, f = tools.read_image_output_slices(LABEL_LIST_TEST)
 
 TEST_length = len(carimages)
 # TEST_length = 1
 
 # carimages = carimages / 255 - 0.5  #TODO here is tricky, double check wit respect to the formats
 # digits.images = carimages.reshape((len(carimages), -1))
 """
  print '\n'
  print "4.print shape of database: ", digits.images.shape  # hy
  digits.images = np.expand_dims(np.array(digits.images), 2).astype(np.float32)
  print "4.1.print shape of database after expansion: ", digits.images.shape  # hy

  digits.target = np.array(cartargets).astype(np.int32)
  digits.target = dense_to_one_hot(digits.target)
  print '\n'
  print "5.print target"
  print digits.target
  """
 confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
 confMat2_TEST = np.zeros((2, 2), dtype=np.float)
 confMat3 = np.zeros((1, n_classes), dtype=np.float)
 count_labels = np.zeros((1, n_classes), dtype=np.float)
 class_probability = np.zeros((1, n_classes), dtype=np.float)
 pred_collect = []
 slices = []
 d = {'key': 'value'}
 for i in range(0, TEST_length, 1):
  # hy:extra Debug
  # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
  
  im = carimages[i]
  
  # im = frame_crop_resize_gray  # Lazy
  
  # from scipy import ndimage   from scipy import misc
  # im = ndimage.gaussian_filter(im, sigma=3)
  # or
  # im = ndimage.uniform_filter(im, size=11) #local mean
  ######################################
  ######################################
  
  im = np.asarray(im, np.float32)
  
  CONF = 0.20
  
  test_image = im
  
  test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
  
  # hy: info
  # print "Image size (wxh):", im.size #hy
  
  # Doing something very stupid here, fix it!
  test_image = im.reshape((-1, im.size))
  
  # print test_image
  # print sess.run(test_image)
  
  test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
  
  batch_xs, batch_ys = test_image, test_labels
  
  # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
  
  output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
  pred_collect.append(output)
  
  # print("Output for external=",output)
  # print output
  output = tools.convert_to_confidence(output)  #
  np.set_printoptions(precision=3)
  
  RES = np.argmax(output)  # hy predicted label
  
  label_target = int(cartargets[i])  # hy ground truth label
  
  # label_pred_str, label_pred_num = tools.convert_result(RES)
  # label_target_str, label_target_num = tools.convert_result(label_target)
  
  predict = int(RES)
  print '\nTestImage', i + 1, ':', f[i]
  # print 'Image name', carimages
  print 'Ground truth label:', LABELS[label_target][:-1], ',  predict:', LABELS[RES][:-1], ', pres:', output[0][
   RES]  # hy
  # print 'output all:', output[0] # hy
  label = label_target
  
  d[f[i]] = output[0][RES]
  
  confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1
  
  count_labels[:, label] = count_labels[:, label] + 1
  
  if predict == label_target:
   label2_TEST = 0
   pred2_TEST = 0
   
   confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
   tools.SAVE_CorrectClassified_Img(f[i], SAVE_CorrectClassified)
  
  
  else:
   label2_TEST = 1
   pred2_TEST = 1
   tools.SAVE_Misclassified_Img(f[i], SAVE_Misclassified)
  
  confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
  tp = confMat2_TEST[0, 0]
  tn = confMat2_TEST[1, 1]
  
  print '\nRank list of predicted results'
  tools.rank_index(output[0], label_target)
  
  print '\nCount correctly classified'
  tools.print_label_title()
  print confMat3
  
  print 'Total labels'
  print count_labels
  
  print 'Proportion of correctly classified'
  for pos in range(0, n_classes, 1):
   if count_labels[:, pos] > 0:
    class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
   else:
    class_probability[:, pos] = 0
  print class_probability
  
  # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
  # print confMat2_TEST
  print '\nTEST general count:'
  print confMat2_TEST
  print 'TEST overall acc:', "{:.3f}".format(tp / TEST_length)
  # print 'pred_collect', pred_collect
  
  
  ###################################################################################
  ## Feature output #################################################################
  ###################################################################################
  
  if TEST_CONV_OUTPUT:
   print '\nTEST feature output:'
   # conv_feature = sess.run(conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
   conv_feature = sess.run(conv2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
   tools.get_feature_map(conv_feature, f, 'conv2')
 else:
  print 'no image got'
 
 # print 'activation list', d
 sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
 print 'sorted', sorted_d
 
 return confMat1_TEST


def EVALUATE_IMAGES(session, num_class, img_list, _labels):  # (eva)
 sess = session
 LABEL_LIST_TEST = img_list
 LABELS = _labels
 n_classes = num_class
 
 ################### active field test part one ################################
 if do_active_fields_test == 1:
  carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
  TEST_length = len(carimages)
  print '1 file', LABEL_LIST_TEST, 'path', LABEL_PATH_TEST, 'len', TEST_length
  # TEST_length = 1
  print 'get active fields'
  row = 0
  col = 0
  test_img_bg = cv2.imread(activation_test_img_name)
  test_img_bg = cv2.resize(test_img_bg, (400, 400))
  overlay = np.zeros([400, 400, 3], dtype=np.uint8)
  test_img_transparent = overlay.copy()
  
  cv2.rectangle(overlay, (0, 0), (400, 400), color=(60, 80, 30, 3))
  alpha = 0.7  # hy: parameter for degree of transparency
  print 'test_img_bg', test_img_bg
  cv2.addWeighted(overlay, alpha, test_img_bg, 1 - alpha, 0, test_img_transparent)
  print 'test_img_transparent', test_img_transparent
  bg = Image.fromarray(test_img_transparent)
  print 'bg done'
 else:
  carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
  TEST_length = len(carimages)
  print '1 file', LABEL_LIST_TEST, 'path', LABEL_PATH_TEST, 'len', TEST_length
 
 if DEBUG == 1 and do_active_fields_test == 1:
  overlay_show = Image.fromarray(overlay)
  overlay_show.save('../1-overlay.jpg')
  bg.save('../1-before.jpg')
 
 ################### active field test part one end ############################
 
 # carimages = carimages / 255 - 0.5  #TODO here is tricky, double check wit respect to the formats
 # digits.images = carimages.reshape((len(carimages), -1))
 """
  print '\n'
  print "4.print shape of database: ", digits.images.shape  # hy
  digits.images = np.expand_dims(np.array(digits.images), 2).astype(np.float32)
  print "4.1.print shape of database after expansion: ", digits.images.shape  # hy

  digits.target = np.array(cartargets).astype(np.int32)
  digits.target = dense_to_one_hot(digits.target)
  print '\n'
  print "5.print target"
  print digits.target
  """
 confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
 confMat2_TEST = np.zeros((2, 2), dtype=np.float)
 confMat3 = np.zeros((1, n_classes), dtype=np.float)
 count_labels = np.zeros((1, n_classes), dtype=np.float)
 class_probability = np.zeros((1, n_classes), dtype=np.float)
 pred_collect = []
 if result_for_table == 0:
  print 'True/False', 'No.', 'Name', 'TargetLabel', 'PredictLabel', 'Precision', 'whole_list', 'Top1', 'Top1_pres', \
   'Top2', 'Top2_pres', 'Top3', 'Top3_pres', 'Top4', 'Top4_pres', 'Top5', 'Top5_pres', 'last', 'last_pres'
 
 for i in range(0, TEST_length, 1):
  # hy:extra Debug
  # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
  
  im = carimages[i]
  # im = frame_crop_resize_gray  # Lazy
  
  im = np.asarray(im, np.float32)
  
  CONF = 0.20
  
  test_image = im
  
  test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
  
  # hy: info
  # print "Image size (wxh):", im.size #hy
  
  # Doing something very stupid here, fix it!
  test_image = im.reshape((-1, im.size))
  
  # print test_image
  # print sess.run(test_image)
  
  test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
  
  batch_xs, batch_ys = test_image, test_lables
  
  # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
  # output = sess.run("Accuracy:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
  output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
  
  # print("Output for external=",output)
  output = tools.convert_to_confidence(output)  #
  np.set_printoptions(precision=3)
  
  RES = np.argmax(output)  # hy predicted label
  
  label_target = int(cartargets[i])  # hy ground truth label
  # label_pred_str, label_pred_num = tools.convert_result(RES)
  # label_target_str, label_target_num = tools.convert_result(label_target)
  
  sorted_vec, prob_all = tools.rank_index(output[0], label_target)
  pred_collect.append(prob_all[0])
  
  ################### active field test part two start ################################
  if do_active_fields_test == 1:
   print 'set up for active fields'
   if col >= 4:
    # print '\ncol is 4'
    col = 0
    row += 1
   if row >= 4:
    # print '\nrow is 4'
    row = 0
   positions = ((col) * 100, (row) * 100, (col + 1) * 100, (row + 1) * 100)  # x0,y0, x1,y1
   col += 1
   
   # define image for obtaining its active fields
   
   # activation_test_img = Image.open('../hintenTest.jpg')
   # activation_test_img = Image.open('../vornTest.jpg')
   # activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_d0_0400_1.jpg')
   # activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_u870_400400.jpg')
   # activation_test_img = Image.open('../Test_Images/hinten_ori1_rz400.jpg')
   # activation_test_img = Image.open('../tmp/resized/oben/oben_t2_1_rz400_u856_400400.jpg')
   # activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_d0_0400_1.jpg')
   # activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_u923_400400.jpg')
   # activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_d0_0400_1.jpg')
   # activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_u825_400400.jpg')
   # activation_test_img_copy = cv2.clone(activation_test_img)
   
   activation_test_img = Image.open(activation_test_img_name)
   
   thresh = float(max(pred_collect) * 0.97)
   print 'thresh', thresh
   if prob_all[0] > thresh:
    # print '\nactive field', positions
    image_crop_part = activation_test_img.crop(positions)
    image_crop_part = image_crop_part.filter(ImageFilter.GaussianBlur(radius=1))
    bg.paste(image_crop_part, positions)
   bg.save('../active_fields.jpg')
  
  ################### active field test end  ################################
  
  
  
  if result_for_table == 1:
   if LABELS[label_target][:-1] == LABELS[RES][:-1]:
    print '\nTestImage', i + 1, f[i], LABELS[label_target][:-1] \
     , LABELS[RES][:-1], prob_all[0],
    for img_i in xrange(n_classes):
     print settings.LABEL_names[sorted_vec[n_classes - 1 - img_i]], prob_all[img_i],
   else:
    print '\nMis-C-TestImage', i + 1, f[i], LABELS[label_target][:-1], \
     LABELS[RES][:-1], prob_all[0],
    for img_i in xrange(n_classes):
     print settings.LABEL_names[sorted_vec[n_classes - 1 - img_i]], prob_all[img_i],
  
  if result_for_table == 0:
   print '\nTestImage', i + 1, ':', f[i]
   # print 'Image name', carimages
   print 'Ground truth label:', LABELS[label_target][:-1], ';  predict:', LABELS[RES][:-1]  # hy
   # print 'Target:', label_target, ';  predict:', RES  # hy
   
   print '\nRank list of predicted results'
   tools.rank_index(output[0], label_target)
  
  label = label_target
  predict = int(RES)
  
  confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1
  
  count_labels[:, label] = count_labels[:, label] + 1
  
  if predict == label_target:
   label2_TEST = 0
   pred2_TEST = 0
   
   confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
   tools.SAVE_CorrectClassified_Img(f[i], SAVE_CorrectClassified)
  
  
  else:
   label2_TEST = 1
   pred2_TEST = 1
   tools.SAVE_Misclassified_Img(f[i], SAVE_Misclassified)
  
  confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
  tp = confMat2_TEST[0, 0]
  tn = confMat2_TEST[1, 1]
 
 # print summary
 print '\n\nCount correctly classified'
 tools.print_label_title()
 print confMat3
 
 print 'Total labels'
 print count_labels
 
 print '\nProportion of correctly classified'
 for pos in range(0, n_classes, 1):
  if count_labels[:, pos] > 0:
   class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
 
 print class_probability
 
 # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
 # print confMat2_TEST
 print '\nTEST general count:'
 print confMat2_TEST
 print 'TEST overall acc:', "{:.3f}".format(tp / TEST_length)
 
 ###################################################################################
 ## Feature output #################################################################
 ###################################################################################
 
 if TEST_CONV_OUTPUT:
  print '\nTEST feature output:'
  
  test_writer = tf.train.SummaryWriter(tensorboard_path + settings.LABELS[label_target], sess.graph)
  wc1 = sess.run("wc1:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  wc2 = sess.run("wc2:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  wd1 = sess.run("wd1:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  w_out = sess.run("w_out:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  
  bc1 = sess.run("bc1:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  bc2 = sess.run("bc2:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  bd1 = sess.run("bd1:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  b_out = sess.run("b_out:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  
  conv_feature = sess.run("conv2:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  
  # conv_feature_2D_batch = tools.get_feature_map(conv_feature,f,'conv2') #get defined conv value, not sure for conv2
  
  # featureImg = sess.run("conv2img:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  
  summary_op = tf.merge_all_summaries()
  test_res = sess.run(summary_op, feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
  
  test_writer.add_summary(test_res, 1)
  # print '2D size',len(conv_feature_2D_batch),'\n',sum(conv_feature_2D_batch[:])
  print 'wc1 shape', wc1.shape, 'wc2:', wc2.shape, 'wd1:', wd1.shape, 'w_out:', w_out.shape
  print 'bc1 shape         ', bc1.shape, 'bc2:', '       ', bc2.shape, 'bd1:    ', bd1.shape, 'b_out:   ', b_out.shape
  print 'pred shape', len(pred_collect)
 
 else:
  print 'no image got'
 return (confMat1_TEST, count_labels, confMat3, class_probability)


def EVALUATE_IMAGES_VAGUE():
 # Testing
 cartargets, f = tools.read_test_images(LABEL_LIST_TEST)
 # print 'cartargets label', cartargets
 TEST_length = 20
 # TEST_length = len(cartargets)
 
 # carimages = carimages / 255 - 0.5  #TODO here is tricky, double check wit respect to the formats
 # digits.images = carimages.reshape((len(carimages), -1))
 
 
 """
  print '\n'
  print "4.print shape of database: ", digits.images.shape  # hy
  digits.images = np.expand_dims(np.array(digits.images), 2).astype(np.float32)
  print "4.1.print shape of database after expansion: ", digits.images.shape  # hy

  digits.target = np.array(cartargets).astype(np.int32)
  digits.target = dense_to_one_hot(digits.target)
  print '\n'
  print "5.print target"
  print digits.target
  """
 
 confMat_m1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)
 confMat_m2_TEST = np.zeros((2, 2), dtype=np.float)
 confMat_m3 = np.zeros((1, n_classes), dtype=np.float)
 count_labels_m = np.zeros((1, n_classes), dtype=np.float)
 class_probability_m = np.zeros((1, n_classes), dtype=np.float)
 
 patch_size = 227
 for i in range(0, TEST_length, 1):
  # hy:extra Debug
  # im = carimages[i]
  # im = frame_crop_resize_gray  # Lazy
  
  
  '''
    #hy: option to use numpy.ndarray, but it cannot use attribute 'crop' of Image (integer) object
    img = cv2.imread(f[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(img, width=patch_size, height=patch_size)
    h_b, w_b = img.shape
    print 'h_b', h_b, ', w_b', w_b
    '''
  
  print 'processing main test image', f[i]
  
  # hy: use integer image: Image, resize
  img = Image.open(f[i]).convert('LA')  # convert to gray
  h_b, w_b = img.size
  # print 'read test image ok', h_b, ', ', w_b
  img = img.resize((patch_size * 2, patch_size * 2), Image.BICUBIC)  # hy:use bicubic
  # h_b, w_b = img.size
  # print 'h_b', h_b, ', w_b', w_b
  
  test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
  test_image = img
  test_image_label = cartargets[i]
  # Doing something very stupid here, fix it!
  # test_image = im.reshape((-1, im.size))
  
  
  # test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  # test_image = test_image / 255 - 0.5  # TODO here is tricky, double check with respect to the formats
  
  slices_rec = prep_image.create_test_slices(test_image, patch_size, test_image_label)
  print 'slices with path received', slices_rec
  slices_len = len(slices_rec)
  
  out_sum = np.zeros((1, n_classes), dtype=np.float)
  out_box = np.zeros((1, n_classes), dtype=np.float)
  
  # batch_xs, batch_ys = im, cartargets
  
  # output_im = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
  
  for j in range(0, slices_len, 1):
   print '\nprocessing slice', j, slices_rec[j]
   # hy read and resize integer object
   # im_s = Image.open(slices_rec[j]) #numpy.ndarray does not have attribute 'crop'
   # im_s = im_s.resize((settings.h_resize, settings.w_resize), Image.BICUBIC)  # hy:use bicubic, resize func reuqires integer object
   # im_s = im_s.convert('LA') #hy convert to gray
   
   # hy read and resize continuous number object
   im_s = cv2.imread(slices_rec[j])  # result is not integer
   im_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
   im_s = imutils.resize(im_s, width=settings.h_resize, height=settings.w_resize)
   
   # hy conver to integer object required for tensor
   im_s = np.asarray(im_s, np.float32)
   
   CONF = 0.20
   
   (sorted_vec, outputsub) = EVALUATE_IMAGE_SLICES(im_s, f, i, sess, cartargets)
   print 'slice', j, 'result', sorted_vec
   print 'Image slice', slices_rec[j]
   outbox = outputsub
   out_sum = out_sum + outputsub[0]
  
  # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
  # print confMat2_TEST
  print '\nTEST general count:'
  
  print out_sum
  print out_sum / slices_len
  outbox[0] = out_sum / slices_len
  
  output_im = tools.rank_index(outbox[0], test_image_label)
  print 'target', test_image_label
  print 'output final prediction', output_im[-1]
  
  RES = int(output_im[-1])
  print 'test_image_label', test_image_label
  
  label = test_image_label
  predict = int(RES)
  
  confMat_m1_TEST[label, predict] = confMat_m1_TEST[label, predict] + 1
  
  count_labels_m[:, test_image_label] = count_labels_m[:, test_image_label] + 1
  
  if int(RES) == int(test_image_label):
   label2_TEST = 0
   pred2_TEST = 0
   
   confMat_m3[:, int(RES)] = confMat_m3[:, int(RES)] + 1
   tools.SAVE_CorrectClassified_Img(f[i], SAVE_CorrectClassified)
  
  
  else:
   label2_TEST = 1
   pred2_TEST = 1
   tools.SAVE_Misclassified_Img(f[i], SAVE_Misclassified)
  
  # print 'Count classified'
  # tools.print_label_title()
  # print confMat1_TEST
  
  confMat_m2_TEST[label2_TEST, pred2_TEST] = confMat_m2_TEST[label2_TEST, pred2_TEST] + 1
  tp = confMat_m2_TEST[0, 0]
  tn = confMat_m2_TEST[1, 1]
  
  print 'Count classified m1 - confusion matrix'
  tools.print_label_title()
  print confMat_m1_TEST
  
  print '\nCount correctly classified -m3'
  tools.print_label_title()
  print confMat_m3
  
  print 'tp,np -m2'
  print confMat_m2_TEST
  print 'Total labels'
  print count_labels_m
  
  print 'Proportion of correctly classified for detailed analysis'  # ok
  if count_labels_m[:, pos] > 0:
   for pos in range(0, n_classes, 1):
    class_probability_m[:, pos] = confMat_m3[:, pos] / count_labels_m[:, pos]
   print class_probability_m
  
  print 'TEST overall acc:', "{:.3f}".format(tp / TEST_length)


def EVALUATE_IMAGE_SLICES(img, f, index, sess, cartargets):  # hy todo change dimension to fit tensorflow
 confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)
 confMat2_TEST = np.zeros((2, 2), dtype=np.float)
 confMat3 = np.zeros((1, n_classes), dtype=np.float)
 count_labels = np.zeros((1, n_classes), dtype=np.float)
 class_probability = np.zeros((1, n_classes), dtype=np.float)
 
 img_s = img
 i = index
 test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
 
 # Doing something very stupid here, fix it!
 test_image = img_s.reshape((-1, img_s.size))
 
 test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
 test_image = test_image / 255 - 0.5  # TODO here is tricky, double check with respect to the formats
 
 batch_xs1, batch_ys1 = test_image, test_lables
 
 output = sess.run(pred, feed_dict={x: batch_xs1, y: batch_ys1, keep_prob: 1.})
 
 # print("Output for external=",output)
 # print output
 output = tools.convert_to_confidence(output)  #
 np.set_printoptions(precision=3)
 
 RES = np.argmax(output)
 
 label_target = int(cartargets[i])
 
 print '\nTestImage', i + 1, ':', f[i]
 
 # print 'Image name', carimages
 print 'Target:', LABELS[label_target][:-1], ';  predict:', LABELS[RES][:-1]  # hy
 # print 'Target:', label_target, ';  predict:', RES  # hy
 
 count_labels[:, label_target] = count_labels[:, label_target] + 1
 
 label = label_target
 predict = int(RES)
 # hy: INFO - print label, predict
 # print 'labels_onehot:', labels_onehot[i, :], '  label=', label
 # print 'score:', scores[i, :]
 # print 'predict:', predict
 # if label == predict:
 confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1
 
 if int(RES) == label_target:
  label2_TEST = 0
  pred2_TEST = 0
  confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
  tools.SAVE_CorrectClassified_Img(f[i], SAVE_CorrectClassified)
 
 else:
  label2_TEST = 1
  pred2_TEST = 1
  tools.SAVE_Misclassified_Img(f[i], SAVE_Misclassified)
 
 # print 'Count classified'
 # tools.print_label_title()
 # print confMat1_TEST
 
 confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
 tp = confMat2_TEST[0, 0]
 tn = confMat2_TEST[1, 1]
 print '\nCount correctly classified'
 tools.print_label_title()
 print confMat3
 
 # print 'Total labels'
 # print count_labels
 
 # print 'Proportion of correctly classified'
 # if count_labels[:,pos] > 0:
 # for pos in range(0, 6, 1):
 #  class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
 # print class_probability
 
 # print '\nRank list of predicted results'
 sorted_vec = tools.rank_index(output[0], label_target)
 
 # return (confMat1_TEST, confMat2_TEST, confMat3, count_labels, class_probability,sorted_vec,output)
 return (sorted_vec, output)


def EVALUATE_WITH_WEBCAM(camera_port, stop):
 # hy: check camera availability
 camera = cv2.VideoCapture(camera_port)
 
 if stop == False:
  # if ckpt and ckpt.model_checkpoint_path:
  # Camera 0 is the integrated web cam on my netbook
  
  # Number of frames to throw away while the camera adjusts to light levels
  ramp_frames = 1
  
  i = 0
  
  while True:  # hy: confirm camera is available
   # Now we can initialize the camera capture object with the cv2.VideoCapture class.
   # All it needs is the index to a camera port.
   print 'Getting image...'
   
   ret, frame = camera.read()
   
   # Captures a single image from the camera and returns it in PIL format
   
   # ret = camera.set(3, 320) #hy use properties 3 and 4 to set frame resolution. 3- w, 4- h
   # ret = camera.set(4, 240)
   
   cv2.waitKey(1)
   # A nice feature of the imwrite method is that it will automatically choose the
   # correct format based on the file extension you provide.
   # cv2.imwrite(file, camera_capture)
   
   ####################################  /////////////////////////////
   
   
   if frame is not None:
    # print 'frame from webcam obtained'
    
    # hy: before continue check if image is read correctly
    # while frame is not None:
    i += 1
    # hy:
    h_frame = frame.shape[0]
    w_frame = frame.shape[1]  # hy: h 1536 x w 2304
    
    # hy: info
    print "h_video and w_video", h_frame, ",", w_frame
    
    # cv2.imshow("ori", frame)
    
    # crop_x1 = int((w_frame - area_step_size_webcam) / 2)
    # crop_y1 = int((h_frame - area_step_size_webcam) / 2)  # 1#200
    
    # crop_x2 = crop_x1 + area_step_size_webcam
    # crop_y2 = int(crop_y1 + area_step_size_webcam * settings.h_resize / settings.w_resize)
    
    crop_y1 = int((h_frame - area_step_size_webcam) / 2)  # 1#200
    crop_x1 = int((w_frame - area_step_size_webcam) / 2)
    
    crop_y2 = crop_y1 + area_step_size_webcam  # hy:define shorter side as unit length to avoid decimal
    crop_x2 = crop_x1 + area_step_size_webcam * settings.w_resize / settings.h_resize
    
    # print "x1,y1,x2,y2", crop_x1, 'x', crop_y1, ',', crop_x2, 'x', crop_y2
    # Crop
    # hy: select suitable values for the area of cropped frame,
    #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
    frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # hy: info
    # print "shape:y1,y2,x1,x2:", crop_y1," ", crop_y2," ", crop_x1," ", crop_x2
    # print "Shape of cropped frame:", frame_crop.shape  #hy: it should be same as the shape of trained images(the input image)
    
    cv2.imshow("frame_cropped", frame_crop)
    
    # Resize
    # frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
    # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
    frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY),
                                            width=settings.w_resize)
    
    # hy:extra Debug
    # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
    
    im = frame_crop_resize_gray  # Lazy
    
    im = np.asarray(im, np.float32)
    
    cv2.imshow("TensorFlow Window", imutils.resize(im.astype(np.uint8), 227))  # hy trial
    
    # Adding noise to the street image #TODO
    # im=add_noise(im,5)
    
    # Bluring the image to help detection #TODO
    # im = cv2.GaussianBlur(im,(5,5),0)
    
    CONF = 0.20
    
    test_image = im
    
    test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
    
    # hy: info
    # print "Image size (wxh):", im.size #hy
    
    # Doing something very stupid here, fix it!
    test_image = im.reshape((-1, im.size))
    # print test_image
    
    test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
    
    # print test_image
    
    
    test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
    
    batch_xs, batch_ys = test_image, test_lables
    
    # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
    
    output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
    
    # print("Output for external=",output)
    # print output
    output = tools.convert_to_confidence(output)
    np.set_printoptions(precision=2)
    print '\nFrame', i
    tools.print_label_title_conf()
    print 'confidence =', output  # hy
    
    RES = np.argmax(output)
    label_pred_str = LABELS[RES][:-1]
    
    # label_pred_str, label_pred_num = tools.convert_result(RES)
    # print 'label_pred_str', label_pred_str
    print 'predicted label:', LABELS[RES][:-1]
    
    if label_pred_str == video_label:
     label2_TEST_Video = 0
     pred2_TEST_Video = 0
     
     name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % i
     tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)
    
    
    else:
     label2_TEST_Video = 1
     pred2_TEST_Video = 1
     
     name_str = settings.Misclassified + "/frame_crop%d.jpg" % i
     tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)
    
    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)
    
    # cv2.putText(frame, "predicted1: " + label_pred_str, org=(w_frame / 10, h_frame / 20),
    #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
    
    cv2.putText(frame, "predicted1: " + label_pred_str, org=(w_frame / 10, h_frame / 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
    
    prob_str = str(output[0][RES])[:4]
    cv2.putText(frame, "prob:" + prob_str, org=(w_frame / 10, h_frame / 8),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
    
    # hy: could be modified to display desired label
    # cv2.putText(frame, LABELS[RES], org=(800, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3 )
    
    
    # cv2.putText(frame, str(video.get(1)), org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    #           color=(0, 255, 0), thickness=1)
    frame_demo = imutils.resize(frame, width=1200)
    
    # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
    cv2.imshow("Demo", frame_demo)
    cv2.waitKey(300)
   
   # TODO add termination condition
   
   print 'no frame retrieved'
 
 del (camera)
 return stop


def EVALUATE_WITH_WEBCAM_track_roi(camera_port):
 frame_index_i = 0
 crop_x1 = 300
 area_step_size = 200
 crop_y1 = 200
 
 # hy: check camera availability
 camera = cv2.VideoCapture(camera_port)
 
 # Read the first frame of the video
 ret, frame = camera.read()
 
 # Set the ROI (Region of Interest). Actually, this is a
 # rectangle of the building that we're tracking
 
 ###############################################################################################
 #   Track
 ###############################################################################################
 
 c, r, w, h = 100, 200, 200, 200
 
 track_window = (c, r, w, h)
 # track_window = (x0, y0, w, h)
 
 # Create mask and normalized histogram
 roi = frame[r:r + h, c:c + w]
 hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
 mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
 
 roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
 
 cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
 
 term_cond = (
  cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)  # hy: TERM_CRITERIA_EPS - terminate iteration condition
 
 # hy: initialization of confmatrix
 confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)
 
 while True:  # hy: confirm camera is available
  # Now we can initialize the camera capture object with the cv2.VideoCapture class.
  # All it needs is the index to a camera port.
  
  print 'Getting image...'
  # Captures a single image from the camera and returns it in PIL format
  ret, frame = camera.read()
  
  # ret = camera.set(3, 320) #hy use properties 3 and 4 to set frame resolution. 3- w, 4- h
  # ret = camera.set(4, 240)
  
  cv2.waitKey(1)
  # A nice feature of the imwrite method is that it will automatically choose the
  # correct format based on the file extension you provide.
  # cv2.imwrite(file, camera_capture)
  
  if ret:
   frame_index_i = frame_index_i + 1
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   print 'hsv done'
   dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
   print 'back project done'
   ret, track_window = cv2.meanShift(dst, track_window, term_cond)
   print 'ret'
   xt, yt, wt, ht = track_window
   # hy info
   print 'xt,yt,wt,ht:', xt, ',', yt, ',', wt, ',', ht
   
   # hy: draw rectangle as tracked window area
   cv2.rectangle(frame, (xt, yt), (xt + wt, yt + ht), 255, 2)
   cv2.putText(frame, 'tracked', (xt - 25, yt - 10), cv2.FONT_HERSHEY_SIMPLEX,
               1, (255, 255, 255), 2, cv2.CV_AA)
   
   cv2.waitKey(100)
   
   ###############################################################################################
   #   evaluate
   ###############################################################################################
   # hy: info
   # print "shape in evaluate:x1,y1:", crop_x1, ',', crop_y1
   
   crop_x1 = xt
   crop_x2 = xt + wt
   crop_y1 = yt
   area_step_size = ht
   crop_y2 = crop_y1 + area_step_size * settings.h_resize / settings.w_resize
   
   # hy: select suitable values for the area of cropped frame,
   #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
   frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
   
   # hy: info
   print "shape after set_video_window:y1, y2, x1, x2:", crop_y1, ',', crop_y2, ',', crop_x1, ',', crop_x2
   
   # hy:
   h_frame = frame.shape[0]
   w_frame = frame.shape[1]  # hy: h 1536 x w 2304
   
   #######################################################################################
   # Evaluate
   #######################################################################################
   
   # hy: info
   # print "shape:y1,y2,x1,x2:", crop_y1," ", crop_y2," ", crop_x1," ", crop_x2
   # print "Shape of cropped frame:", frame_crop.shape  #hy: it should be same as the shape of trained images(the input image)
   
   cv2.imshow("frame_cropped", frame_crop)
   
   # Resize
   # frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
   # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
   frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY),
                                           width=settings.w_resize)
   
   # hy:extra Debug
   # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
   
   im = frame_crop_resize_gray  # Lazy
   
   im = np.asarray(im, np.float32)
   
   cv2.imshow("TensorFlow Window", imutils.resize(im.astype(np.uint8), 200))
   
   # Adding noise to the street image #TODO
   # im=add_noise(im,5)
   
   # Bluring the image to help detection #TODO
   # im = cv2.GaussianBlur(im,(5,5),0)
   
   CONF = 0.20
   
   test_image = im
   
   test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
   
   # hy: info
   # print "Image size (wxh):", im.size #hy
   
   # Doing something very stupid here, fix it!
   test_image = im.reshape((-1, im.size))
   # print test_image
   
   test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
   
   # print test_image
   
   
   test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
   
   batch_xs, batch_ys = test_image, test_lables
   
   # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
   
   output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
   
   # print("Output for external=",output)
   # print output
   output = tools.convert_to_confidence(output)
   np.set_printoptions(precision=2)
   print '\nFrame', frame_index_i
   tools.print_label_title_conf()
   print 'confidence =', output  # hy
   
   RES = np.argmax(output)
   if int(RES) == video_label:
    label2_TEST_Video = 0
    pred2_TEST_Video = 0
    
    name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % frame_index_i
    tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)
   
   
   else:
    label2_TEST_Video = 1
    pred2_TEST_Video = 1
    
    name_str = settings.Misclassified + "/frame_crop%d.jpg" % frame_index_i
    tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)
   
   cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)
   
   cv2.putText(frame, "predicted1: " + LABELS[RES], org=(w_frame / 10, h_frame / 20),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
   
   # cv2.putText(frame, "predicted2: " + LABELS[RES], org=(w_frame / 10, h_frame / 20),
   #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0, 255, 0), thickness=5)
   
   output_display = str(output[0][RES])[:4]
   cv2.putText(frame, "prob:" + output_display, org=(w_frame / 10, h_frame / 8),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
   
   # cv2.putText(frame, "predicted1: " + LABELS[RES] + ", prob:" + output[RES], org=(w_frame / 6, h_frame / 10),
   #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)
   
   
   # hy: could be modified to display desired label
   # cv2.putText(frame, LABELS[RES], org=(800, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3 )
   
   
   # cv2.putText(frame, str(video.get(1)), org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
   #           color=(0, 255, 0), thickness=1)
   frame_demo = imutils.resize(frame, width=1200)
   
   # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
   cv2.imshow("Demo", frame_demo)
   cv2.waitKey(300)
  
  else:
   print 'no frame retrieved'
   break
  
  # hy TODO add termination condition
 
 del (camera)


def Evaluate_VIDEO_track_roi(VIDEO_FILE):
 video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
 # cv2.waitKey(10)
 
 
 video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
 # hy: propID=1 means 0-based index of the frame to be decoded/captured next
 
 if not video.isOpened():
  print "cannot find or open video file"
  exit(-1)
 
 # Read the first frame of the video
 ret, frame = video.read()
 
 # Set the ROI (Region of Interest). Actually, this is a
 # rectangle of the building that we're tracking
 
 ###############################################################################################
 #   Track
 ###############################################################################################
 
 c, r, w, h = 600, 450, 600, 600
 
 track_window = (c, r, w, h)
 # track_window = (x0, y0, w, h)
 
 # Create mask and normalized histogram
 roi = frame[r:r + h, c:c + w]
 hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
 mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
 
 roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
 
 cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
 
 term_cond = (
  cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)  # hy: TERM_CRITERIA_EPS - terminate iteration condition
 
 # hy: initialization of confmatrix
 confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)
 video_frame_i = 0
 
 while True:
  
  ret, frame = video.read()
  if ret:
   video_frame_i = video_frame_i + 1
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
   ret, track_window = cv2.meanShift(dst, track_window, term_cond)
   
   xt, yt, wt, ht = track_window
   # hy info
   # print 'xt,yt,wt,ht:', xt, ',', yt, ',', wt, ',' , ht
   
   # hy: draw rectangle as tracked window area
   cv2.rectangle(frame, (xt, yt), (xt + wt, yt + ht), 255, 2)
   cv2.putText(frame, 'tracked', (xt - 25, yt - 10), cv2.FONT_HERSHEY_SIMPLEX,
               1, (255, 255, 255), 2, cv2.CV_AA)
   
   cv2.waitKey(500)
   
   ###############################################################################################
   #   evaluate
   ###############################################################################################
   # hy: info
   # print "shape in evaluate:x1,y1:", crop_x1, ',', crop_y1
   
   crop_x1 = xt
   crop_x2 = xt + wt
   crop_y1 = yt
   area_step_size = ht
   crop_y2 = crop_y1 + area_step_size * settings.h_resize / settings.w_resize
   
   # hy: select suitable values for the area of cropped frame,
   #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
   frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
   
   # hy: info
   # print "shape after set_video_window:y1, y2, x1, x2:", crop_y1,',', crop_y2, ',', crop_x1, ',',  crop_x2
   
   
   # hy:
   h_frame = frame.shape[0]
   w_frame = frame.shape[1]  # hy: h 1536 x w 2304
   
   # hy: info
   # print "Shape of cropped frame:", frame_crop.shape  #hy: it should be same as the shape of trained images(the input image)
   
   cv2.imshow("frame_cropped", frame_crop)
   
   # Resize
   # frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
   # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
   frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=settings.w_resize)
   
   # hy:extra Debug
   # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
   
   im = frame_crop_resize_gray  # Lazy
   
   im = np.asarray(im, np.float32)
   
   cv2.imshow("TensorFlow Window", imutils.resize(im.astype(np.uint8), 200))
   
   # Adding noise to the street image #TODO
   # im=add_noise(im,5)
   
   # Bluring the image to help detection #TODO
   # im = cv2.GaussianBlur(im,(5,5),0)
   
   CONF = 0.20
   
   test_image = im
   
   test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
   
   # hy: info
   # print "Image size (wxh):", im.size #hy
   
   # Doing something very stupid here, fix it!
   test_image = im.reshape((-1, im.size))
   # print test_image
   
   test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
   
   # print test_image
   
   
   test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
   
   batch_xs, batch_ys = test_image, test_lables
   
   # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
   
   output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
   
   # print("Output for external=",output)
   # print output
   output = tools.convert_to_confidence(output)
   np.set_printoptions(precision=2)
   
   print '\nFrame', video_frame_i
   tools.print_label_title_conf()
   print 'confidence =', output  # hy
   
   RES = np.argmax(output)
   print "argmax =", np.argmax(output)  # hy
   
   print "label, predict =", video_label, ', ', RES  # hy
   if int(RES) == video_label:
    label2_TEST_Video = 0
    pred2_TEST_Video = 0
    
    name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % video_frame_i
    tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)
   
   else:
    label2_TEST_Video = 1
    pred2_TEST_Video = 1
    
    name_str = settings.Misclassified + "/frame_crop%d.jpg" % video_frame_i
    tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)
   
   confMat2_TEST_Video[label2_TEST_Video, pred2_TEST_Video] = confMat2_TEST_Video[
                                                               label2_TEST_Video, pred2_TEST_Video] + 1
   
   cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)
   cv2.putText(frame, "predicted: " + LABELS[RES], org=(w_frame / 3, h_frame / 10),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=4)
   
   cv2.putText(frame, str(video.get(1)), org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
               color=(0, 255, 0), thickness=1)
   frame_demo = imutils.resize(frame, width=1200)
   
   # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
   cv2.imshow("Demo", frame_demo)
   cv2.waitKey(300)
  
  
  
  
  # hy: other options - control to move ROI downwards, then to the right
  # crop_y1 = crop_y1 + area_step_size/50
  
  # if crop_y2+area_step_size >= frame.shape[0]:
  # crop_y1 = 0
  # crop_x1 = crop_x1 + 200
  # if crop_x2+area_step_size >= frame.shape[1]:
  ##crop_x1 = 0
  # break
  
  else:
   print 'no frame retrieved'
   break  # hy added
  
  tp = confMat2_TEST_Video[0, 0]
  tn = confMat2_TEST_Video[1, 1]
  # print confMat2_TEST_Video
  # print 'tp, tn, total number of test images:', tp, ', ', tn, ', ', tp + tn
  print confMat2_TEST_Video
  print 'TEST acc:', "{:.4f}".format(tp / (tp + tn))
  cv2.waitKey(100)
  if cv2.waitKey(1) & 0xFF == ord('q'):  # hy:press key-q to quit
   break
  
  ###############################################################################################
  
  # cv2.imshow('Tracking', frame)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   break
  # else:
  # print 'no frame received for tracking'
  # break


def EVALUATE_VIDEO(VIDEO_FILE):
 video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
 # cv2.waitKey(10)
 
 video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
 # hy: propID=1 means 0-based index of the frame to be decoded/captured next
 
 # video.open(VIDEO_FILE)
 
 # hy: for debug
 if not video.isOpened():
  print "cannot find or open video file"
  exit(-1)
 
 ## Reading the video file frame by frame
 
 # hy: initialization of confmatrix
 confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)
 video_frame_i = 0
 while True:
  video_frame_i += 1
  ret, frame = video.read()
  if ret:
   
   # hy:
   h_frame = frame.shape[0]
   w_frame = frame.shape[1]  # hy: h 1536 x w 2304
   
   # hy: info
   # print "h_video and w_video", h_resize, ",", w_resize
   
   # cv2.imshow("ori", frame)
   # print "frame size hxw", frame.shape[0]," ", frame.shape[1]
   
   crop_x2 = crop_x1 + area_step_size
   # crop_y2 = (crop_y1 + (crop_x2 - crop_x1)) * settings.h_resize / settings.w_resize
   crop_y2 = crop_y1 + area_step_size * settings.h_resize / settings.w_resize
   
   # Crop
   # frame_crop = frame[350:750, 610:1300] #hy: ori setting for w24xh42
   
   # hy: select suitable values for the area of cropped frame,
   #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
   frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
   
   # hy: info
   print "shape:y1,y2,x1,x2:", crop_y1, ", ", crop_y2, ", ", crop_x1, ", ", crop_x2
   # print "Shape of cropped frame:", frame_crop.shape  #hy: it should be same as the shape of trained images(the input image)
   
   cv2.imshow("frame_cropped", frame_crop)
   
   # Resize
   # frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
   # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
   frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=settings.w_resize)
   
   # hy:extra Debug
   # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape
   
   im = frame_crop_resize_gray  # Lazy
   
   im = np.asarray(im, np.float32)
   
   cv2.imshow("TensorFlow Window", imutils.resize(im.astype(np.uint8), 227))
   
   # Adding noise to the street image #TODO
   # im=add_noise(im,5)
   
   # Bluring the image to help detection #TODO
   # im = cv2.GaussianBlur(im,(5,5),0)
   
   
   CONF = 0.20
   
   test_image = im
   
   test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
   
   # hy: info
   # print "Image size (wxh):", im.size #hy
   
   # Doing something very stupid here, fix it!
   test_image = im.reshape((-1, im.size))
   # print test_image
   
   test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
   
   # print test_image
   
   test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
   
   batch_xs, batch_ys = test_image, test_lables
   
   # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
   
   output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
   
   # print("Output for external=",output)
   # print output
   output = tools.convert_to_confidence(output)
   np.set_printoptions(precision=2)
   
   print '\nFrame', video_frame_i
   tools.print_label_title_conf()
   print 'confidence =', output  # hy
   
   RES = np.argmax(output)
   print "argmax =", np.argmax(output)  # hy
   
   label_pred_str = LABELS[RES][:-1]
   
   # hy: qfor sub-classes
   # label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
   # RES_sub_to_face = class_label #hy added
   
   print "label, predict =", video_label, ', ', RES  # hy
   
   if label_pred_str == video_label:
    label2_TEST_Video = 0
    pred2_TEST_Video = 0
    
    name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % video_frame_i
    tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)
   
   else:
    label2_TEST_Video = 1
    pred2_TEST_Video = 1
    
    name_str = settings.Misclassified + "/frame_crop%d.jpg" % video_frame_i
    tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)
   
   confMat2_TEST_Video[label2_TEST_Video, pred2_TEST_Video] = confMat2_TEST_Video[
                                                               label2_TEST_Video, pred2_TEST_Video] + 1
   
   # Make a little demonstration (hy:static window version)
   # hy: showing evaluation result identified class on video
   # if RES == 0 or RES == 2:
   #  cv2.rectangle(frame,(610, 350), (1300, 750), color=(0, 255, 0), thickness=20)
   #  cv2.putText(frame, 'Available', org=(800, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0),thickness=4)
   # else:
   #  cv2.rectangle(frame,(610, 350), (1300, 750), color=(0, 0, 255), thickness=20)
   #  cv2.putText(frame, 'Occupied', org=(800, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255), thickness=4)
   
   # hy: TODO adapt to current app
   # if RES == 0 or RES == 2:
   #  cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=20)
   #  cv2.putText(frame, 'Available', org=(800, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
   #              color=(0, 255, 0), thickness=4)
   # else:
   #  cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 0, 255), thickness=20)
   #  cv2.putText(frame, 'Occupied', org=(800, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
   #              color=(0, 0, 255), thickness=4)
   
   
   cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)
   cv2.putText(frame, "predicted: " + label_pred_str, org=(w_frame / 3, h_frame / 10),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=4)
   
   # hy: could be modified to display desired label
   # cv2.putText(frame, label_pred_str, org=(800, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3 )
   
   
   cv2.putText(frame, str(video.get(1)), org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
               color=(0, 255, 0), thickness=1)
   # cv2.putText(frame, label_pred_str, org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
   #            color=(0, 255, 0), thickness=1)
   frame_demo = imutils.resize(frame, width=1200)
   
   # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
   cv2.imshow("Demo", frame_demo)
   cv2.waitKey(300)
  
  
  # hy: other options - control to move ROI downwards, then to the right
  # crop_y1 = crop_y1 + area_step_size/50
  
  # if crop_y2+area_step_size >= frame.shape[0]:
  # crop_y1 = 0
  # crop_x1 = crop_x1 + 200
  # if crop_x2+area_step_size >= frame.shape[1]:
  ##crop_x1 = 0
  # break
  
  else:
   print 'no frame retrieved'
   break  # hy added
  
  tp = confMat2_TEST_Video[0, 0]
  tn = confMat2_TEST_Video[1, 1]
  # print confMat2_TEST_Video
  # print 'tp, tn, total number of test images:', tp, ', ', tn, ', ', tp + tn
  print confMat2_TEST_Video
  print 'TEST acc:', "{:.4f}".format(tp / (tp + tn))
  
  if cv2.waitKey(1) & 0xFF == ord('q'):  # hy:press key-q to quit
   break


def confusion_matrix(labels_onehot, scores, normalized=True):
 n_samples, n_class = scores.shape
 print 'n_samples for validation:', n_samples
 conf_matrix = np.zeros((n_class, n_class), dtype=np.float32)
 conf_matrix_2 = np.zeros((2, 2), dtype=np.float32)
 
 for i in range(0, n_samples):
  label = np.argmax(labels_onehot[i, :])
  predict = np.argmax(scores[i, :])
  # hy: INFO - print label, predict
  # print 'labels_onehot:', labels_onehot[i, :], '  label=', label
  # print 'score:', scores[i, :]
  # print 'predict:', predict
  conf_matrix[label, predict] = conf_matrix[label, predict] + 1
  
  # Mapping labels
  '''
    if label == 0 or label == 2:
      label2 = 0
    else:
      label2 = 1

    if predict == 0 or predict == 2:
      predict2 = 0
    else:
      predict2 = 1
    '''
  
  #################################################################################################################
  # hy: adapt to lego classes
  # hy: use it count corrected predict
  
  
  # print label2, predict2
  if label == predict:  # hy: true positive
   # hy: conf_matrix_2 true positive index 0,0
   label2 = 0
   predict2 = 0
  
  
  
  else:
   # hy: conf_matrix_2 true positive index 1,1
   label2 = 1
   predict2 = 1
  
  #################################################################################################################
  
  conf_matrix_2[label2, predict2] = conf_matrix_2[label2, predict2] + 1.0
 
 # hy: confusion matrix
 # [  tp      fn]
 # [  fp      tn]
 # tp: count label=predict / total
 # tn: label!=predict
 # fp: 1-tp
 # fn: 1-tn
 
 if normalized:
  for i in range(0, n_class):
   conf_matrix[i, :] = conf_matrix[i, :] / np.sum(conf_matrix[i, :])
 
 return conf_matrix, conf_matrix_2


# def dense_to_one_hot(labels_dense, num_classes=n_classes):
def dense_to_one_hot(labels_dense, num_classes):
 """Convert class labels from scalars to one-hot vectors."""
 num_labels = labels_dense.shape[0]
 index_offset = np.arange(num_labels) * num_classes
 labels_one_hot = np.zeros((num_labels, num_classes))
 labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
 if DEBUG == 1:
  print 'one_hot_vector:', labels_one_hot[0]
 return labels_one_hot


# Implementing softmax function on the DL output scores, adopted only for 2 classes
# hy: for final output layer using softmax classification
def convert_to_confidence(scores):
 h, w = scores.shape
 output = np.zeros((h, w), dtype=np.float32)
 sum = np.zeros((h, 1), dtype=np.float32)
 # if sum != 0:
 for i in range(0, w):
  sum[:, 0] = sum[:, 0] + np.exp(scores[:, i])
 # print 'sum i =', sum[:, 0]
 for i in range(0, w):
  # print 'sum out =', sum[:, 0]
  output[:, i] = np.exp(scores[:, i]) / sum[:, 0]
 # class0=math.exp(scores[0,0])/(math.exp(scores[0,1])+math.exp(scores[0,0]))
 #    class1=math.exp(scores[0,1])/(math.exp(scores[0,1])+math.exp(scores[0,0]))
 #  output=[class0, class1]
 # else:
 #  print 'sum is 0'
 return output


# Adds noise to gray level images, nomalizes the image again
def add_noise(img, noise_level):
 img = img.astype(np.float32)
 h = img.shape[0]
 w = img.shape[1]
 img_noised = img + np.random.rand(h, w) * noise_level
 img_noised = (img_noised / np.max(img_noised)) * 255
 # img_noised=img_noised.astype(np.uint8)
 return img_noised

'''
# Create model
def conv2d(img, w, b, k):
 return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'), b))


def max_pool(img, k):
 return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
'''

# remove old files in tensorflow folder
if RETRAIN:
 cmd = 'rm -rf ' + tensorboard_path + '/*'
 os.system(cmd)

if Graph_3conv == 1:
  import Graph_3conv as g3
  print 'import graph3'
  n_hidden, learning_rate, dropout, dropout_1s, optimizer_type, pred, x, y, \
  keep_prob, optimizer, accuracy, cost, summary_op = g3.define_model()

if Graph_3conv_same_dropout == 1:
 import Graph_3conv_uni

if Graph_4conv == 1:
 import Graph_4conv as g4
 print 'import graph4'
 n_hidden, learning_rate, dropout, dropout_1s, optimizer_type, pred, x, y, \
 keep_prob, optimizer, accuracy, cost, summary_op = g4.define_model()

################################################
# hy: display jpeg image via iPython for terminal and Qt-based, web-based notebook
# Image of IPython package will cause conflict with Image for Python
# error like 'Image does not have attribute fromarray
# from cStringIO import StringIO
# from IPython.display import clear_output, Image, display
# def showarray(a, fmt='jpeg'):
#  a = np.uint8(np.clip(a, 0, 255))
#  f = StringIO()
#  PIL.Image.fromarray(a).save(f, fmt)
#  display(Image(data=f.getvalue()))
# use ipython
# img = np.float32(PIL.Image.open('../Data/tmp_rz82_d8_8.jpg'))
# showarray(img)


class Logger(object):
 def __init__(self):
  self.terminal = sys.stdout
  from datetime import datetime
  str_log = optimizer_type + str(n_hidden) + '_' + 'Rate' + str(learning_rate) + '_' + arch_str
  self.log = open(datetime.now().strftime('../logs/log_%Y_%m_%d_%H_%M' + str_log + '.log'), "a")

 def write(self, message):
  self.terminal.write(message)
  self.log.write(message)

 def flush(self):
  # this flush method is needed for python 3 compatibility.
  # this handles the flush command by doing nothing.
  # you might want to specify some extra behavior here.
  pass


if log_on and (RETRAIN or CONTINUE_TRAIN or TEST_with_Video):
 sys.stdout = Logger()

 # hy:customized tensor model name
model_path_str = '../logs/model_' + optimizer_type + str(n_hidden) + '_h' + \
                 str(settings.h_resize) + '_w' + str(settings.w_resize) \
                 + '_c' + str(n_classes)  # hy include specs of model


##################### TRAINING ####################################
if RETRAIN or CONTINUE_TRAIN:

 try:
  total_images, digits, carimages, cartargets, f, val2_digits, val2_images, val2_targets, val2_f = tools.import_data()
  train_size = int(total_images * TrainingProp)
  print 'train size', train_size
  batch_size = 200
  # batch_size = int(train_size / n_classes * 2)# *2
  
  
  print 'batch size', batch_size
  val1_batch_xs, val1_batch_ys = digits.images[train_size + 1:total_images - 1], \
                                 digits.target[train_size + 1:total_images - 1]
  
  val2_batch_xs, val2_batch_ys = val2_digits.images[0:len(val2_images) - 1], \
                                 val2_digits.target[0:len(val2_images) - 1]  # hy: use calc size
 except:
  print 'Check if file is created correctly. Setting an array element with a sequence.'
 
 # Launch the graph
 with tf.Session() as sess:
  saver = tf.train.Saver()  # hy:
  
  if RETRAIN:
   # Initializing the variables
   init = tf.initialize_all_variables()
   sess.run(init)
  if CONTINUE_TRAIN:
   #sess,saver = tools.load_classifier_model(sess, '../logs/', classifier_model=classifier_model)
   #'''
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../logs/")
   print 'ckpt', ckpt, 'ckpt path', ckpt.model_checkpoint_path
   
   if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print "Continue to train with ", ckpt.model_checkpoint_path
   else:
    print 'not found model'
   #'''
  
  elapsed_time = time.time() - start_time
  
  print 'Total elapsed time3:', "{:.2f}".format(elapsed_time), 's'
  
  # hy: added to display all results in one graph
  train_writer = tf.train.SummaryWriter(tensorboard_path + '/train', sess.graph)
  validation_writer = tf.train.SummaryWriter(tensorboard_path + '/vali', sess.graph)
  #test_writer = tf.train.SummaryWriter(tensorboard_path + '/test', sess.graph)
  
  # from datetime import datetime
  # tensorboard_path = '../Tensorboard_data/sum107/'+str(datetime.now())+'/'
  # summary_writer = tf.train.SummaryWriter(tensorboard_path, graph_def=sess.graph_def)
  
  if RETRAIN:
   step = 1
  if CONTINUE_TRAIN:
   step = current_step
  
  # hy register finished class learning
  acc_pre = 0
  # Keep training until reach max iterations
  while step < training_iters and not set_STOP:
   # Only a part of data base is used for training, the rest is used for validation
   # batch_xs, batch_ys = digits.images[0:850], digits.target[0:850]
   for batch_step in xrange(int(train_size / batch_size)):
    batch_xs, batch_ys = digits.images[int(batch_step * batch_size):(batch_step + 1) * batch_size - 1], \
                         digits.target[batch_step * batch_size:(batch_step + 1) * batch_size - 1]
    print 'batch', batch_step, ', from', int(batch_step * batch_size), 'to', (batch_step + 1) * batch_size - 1
    ## Training  ####################################################################
    
    # hy:define training size - batch size 75% of data base size,
    # training_size = int(total_images * TrainingProp)
    # print 'training size:', training_size
    # batch_xs, batch_ys = digits.images[0:training_size], digits.target[0:training_size]
    
    # batch_xs, batch_ys = digits.images[851:1050], digits.target[851:1050]
    
    # Fit training using batch data
    # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    try:
     sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
     # Calculate batch accuracy
     train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_1s})
     
     # Calculate batch loss
     loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_1s})
    except:
     print '\n[Hint] if error occurs, check data input path, settings label size, \ninput tensor size, input for densel' \
           'is multiplication of the dimension sizes (HxWxD) of previous layer and view size for conv layers, \notherwise, the input tensor size must be changed'
    
    if step % display_step == 0:
     # elapsed_time = time.time() - start_time
     # print 'Up to now elapsed time:', "{:.2f}".format(elapsed_time / 60), 'min'
     
     # if step % time_display_interval == 0:
     elapsed_time = time.time() - start_time
     print 'Up to now elapsed time:', "{:.2f}".format(elapsed_time / 60), 'min'
     
     print "\nIter " + str(step) + '-' + str(batch_step) + ", Minibatch Loss= " + "{:.6f}".format(
      loss) + ", Training Accuracy= " \
           + "{:.4f}".format(train_acc)
     

     ## Validation  ####################################################################

     val1_acc = sess.run(accuracy, feed_dict={x: val1_batch_xs, y: val1_batch_ys, keep_prob: dropout_1s})
     print "Validation accuracy=", "{:.4f}".format(val1_acc)


     ## Test        ####################################################################
     new_graph = tf.Graph()
     with tf.Session(graph=new_graph) as sess2:
      # method 2 must initial sess
      # hy: load saved model with values
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../tensor_model_sum/")  # ../backupModel/
      classifier_model = "../tensor_model_sum/model_GD360_h184_w184_c6_5_3conv_O1.0_R0.7_U0.75_V1.0_5_0.7-10401.meta"
      saver2 = tf.train.import_meta_graph(classifier_model)

      ckpt.model_checkpoint_path = classifier_model[:-5]

      if ckpt and ckpt.model_checkpoint_path:
       print '\n################################\nsecond model'
       saver2 = tf.train.Saver()
       saver2.restore(sess2, ckpt.model_checkpoint_path)
      else:
       print 'not found model'


      for test_i in xrange(len(n_classes)):  #val2_batch_xs, val2_batch_ys
       output = sess.run("pred:0", feed_dict={"x:0": val2_batch_xs, "y:0": val2_batch_ys, "keep_prob:0": dropout_1s})
       output = tools.convert_to_confidence(output)  #
       np.set_printoptions(precision=3)

       RES = np.argmax(output)  # hy predicted label
       # print("Output for external=",output)

       label_target = int(val2_batch_ys[test_i])  # hy ground truth label



     # print "Targets:", batch_ys
     # output = tools.convert_to_confidence(output)
     
     # hy: changed normalized=False to True
     confMat, confMat2 = confusion_matrix(val2_batch_ys, output, normalized=True)
     
     np.set_printoptions(precision=2)  # hy: set display floating point, not changing real value
     
     # print 'Iter:', str(step), ' confMat'
     tools.print_label_title()
     
     print confMat  # hy: shape n_classes x n_classes
     
     print "\nconfinksMat2"
     print confMat2
     
     # print np.sum(confMat)
     # print output
     
     
     # summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
     # summary_writer.add_summary(summary_str, step)
     
     # hy: added to display all results in one graph
     train_res = sess.run(summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_1s})
     train_writer.add_summary(train_res, step)
     
     val1_res = sess.run(summary_op, feed_dict={x: val1_batch_xs, y: val1_batch_ys, keep_prob: dropout_1s})
     validation_writer.add_summary(val1_res, step)
     
     val2_res = sess.run(summary_op, feed_dict={x: val2_batch_xs, y: val2_batch_ys, keep_prob: dropout_1s})
     test_writer.add_summary(val2_res, step)
     
     # Retrain hy: control stop
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
     
     if save_all_models == 1:
      saver.save(sess, save_path=model_path_str + 'all_' + str(batch_step) + '_' + str(round(val2_acc, 2)),
                 global_step=step)  # hy: added. It saves both all variables and GRAPH
     
     if (num_of_classified_classes > 1) or loss < last_loss or val2_acc > last_best_test_acc:
      if loss < last_loss:
       last_loss = loss
      if val2_acc > last_best_test_acc:
       last_best_acc = val2_acc
       test_acc_str = str(round(last_best_acc, 2))
      # Save the model
      if num_of_classified_classes > 2 and sum_col_min < 1.01 and val2_acc > last_best_test_acc - 0.001 \
          and loss < 0.09 and val2_acc > 0.7:
       for p in xrange(num_of_classified_classes):
        name_ext += '_' + max_classes_names[p] + str(max_classes_pres[p])
       name_ext += '_' + str(batch_step) + '_' + str(round(val2_acc, 2))
       print 'save model', name_ext
       # saver.save(sess, save_path=model_path_str + '_I', global_step=step)  # hy: it only saves variables
       saver.save(sess, save_path=model_path_str + '_' + str(batch_step) + '_' + arch_str + name_ext,
                  global_step=step)  # hy: added. It saves GRAPH
       
       cmd = 'mv ../logs/model*' + arch_str + '* ' + tensor_model_sum_path
       os.system(cmd)
       cmd = 'rm ../logs/model*'
       os.system(cmd)
     
     if val2_acc > 0.2 and (float(val2_loss / loss) > stop_loss
                            or float(train_acc / val2_acc) > stop_acc_diff) \
         or float(loss / last_loss) > stop_train_loss_increase_rate:
      if float(val2_loss / loss) > stop_loss:
       print 'Overfitting: loss gap'
      if float(train_acc / val2_acc) > stop_acc_diff:
       print 'Training will be terminated because of overfitting.'
      if float(loss / last_loss) > stop_train_loss_increase_rate:
       print 'Training will be terminated because of increasing loss'
      
      set_STOP = True
      val2_acc = 1
      
      imgNum = len([name for name in os.listdir(settings.data + settings.LABELS[0]) if
                    os.path.isfile(os.path.join(settings.data + settings.LABELS[0], name))])
      
      # if (acc - val2_acc) > 0.1 and imgNum < 3* settings.maxNumSaveFiles: #hy: activate random rotation
      if val2_acc > act_min and val2_acc < act_max and imgNum < 2.3 * settings.maxNumSaveFiles:  # hy: activate random rotation
       # rotation_angle = np.random.rand(0, 180) #hy: not working
       rotation_angle = randint(15, 170)
       noise_level = 0.01 * randint(1, 2)
       if imgNum > 2 * settings.maxNumSaveFiles:
        tools.REMOVE_online_Data(step)
       prep_image.rotateflipImg(rotation_angle, 0, noise_level, step)  # hy: angle,flipX,noise_level,step
       add_data = 1
      # training_size = int(total_images * TrainingProp)
      # batch_xs, batch_ys = digits.images[0:training_size], digits.target[0:training_size]
      
      '''
            #hy try to adjust learning rate automatically, unsupervised learning
            if acc < acc_pre * 0.7:
              learning_rate = learning_rate * 1.1
              print 'increase learning_rate:', learning_rate
              optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
              sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                             keep_prob: dropout})  # hy should not restart here, not interative any more
            elif acc_pre <> 0 and acc > 2.6 * acc_pre:
              learning_rate = learning_rate * 0.1
              print 'reduce learning_rate:', learning_rate
              optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
              sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                             keep_prob: dropout})  # hy should not restart here, not interative any more

            acc_pre = acc
            '''
      
      if add_data == 1:
       print 'update data list'
       tools.prepare_list(LABEL_LIST, LABEL_PATH)  # hy: update file_list
       total_images, digits, carimages, cartargets, f, val2_digits, val2_images, val2_targets, val2_f = tools.import_data()
       training_size = int(total_images * TrainingProp)
      
      # total_images = len(carimages)
      # hy:define training size - batch size 75% of data base size,
      # batch_xs, batch_ys = digits.images[0:training_size], digits.target[0:training_size]
      # cannot change add_data to 0
   
   step += 10
  
  print "\nOptimization Finished!"

#####################################################################################################
##################                 TEST with Video                        ###########################
#####################################################################################################

if TEST_with_Video:
 with tf.Session() as sess:
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
  if ckpt and ckpt.model_checkpoint_path:
   saver.restore(sess, ckpt.model_checkpoint_path)
   print "Evaluation with video, model", ckpt.model_checkpoint_path
  else:
   print 'not found model'
  
  print 'Test with video starting ...'
  # for video_index in xrange(1):
  video_list = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
  
  for video_index in xrange(len(video_list)):
   TestFace = settings.LABELS[0][
              :-1]  # only one 'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
   # TestFace = video_list[video_index][:-1] # all # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
   print 'Test face:', TestFace
   # TestFace = settings.LABELS[video_index][:-1] #'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
   VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
   
   # hy: info
   # print "shape after set_video_window:x1,y1:", crop_x1, ", ", crop_y1
   
   # track_frame = track_roi(VIDEO_FILE)
   # Evaluate_VIDEO_track_roi(VIDEO_FILE)
   
   EVALUATE_VIDEO(VIDEO_FILE)
   print 'test face:', TestFace, 'done\n'
  
  # TestFace = 'vorn'
  
  # VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = set_video_window(TestFace, video_window_scale)
  # EVALUATE_VIDEO(VIDEO_FILE)
  # print 'test face:', TestFace, 'done\n'
  
  # hy: another option - automatically move ROI downwards, then to the right
  # crop_y1 = crop_y1 + area_step_size/50
  # if crop_y2+area_step_size >= frame.shape[0]:
  # crop_y1 = 0
  # crop_x1 = crop_x1 + 200
  # if crop_x2+area_step_size >= frame.shape[1]:
  ##crop_x1 = 0
  # break
#####################################################################################################
##hy: ################                   TEST with IMAGES  (eva)              #######################
#####################################################################################################
init = tf.initialize_all_variables()  # hy

if TEST_with_Images:
 # hy: use a previous model
 # hy: load model at checkpoint
 # method 1
 with tf.Session() as sess:
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
  if ckpt and ckpt.model_checkpoint_path:
   print 'Test with images with', ckpt.model_checkpoint_path
   saver.restore(sess, ckpt.model_checkpoint_path)
   print 'Test list for image test', LABEL_LIST_TEST, 'labels', LABELS
   confMat1_TEST_i, count_labels, confMat3, class_probability = EVALUATE_IMAGES(sess, 6, LABEL_LIST_TEST, LABELS)
  
  # filename = ".".join([tf.latest_checkpoint('/tmp/my-tensor-model.meta'), "meta"])
  # tf.train.import_meta_graph(filename)
  # hparams = tf.get_collection("hparams")
  # print 'hyper parameters:', hparams
  
  print 'Count classified in each class for detailed analysis'
  tools.print_label_title()
  print confMat1_TEST_i
 
 ######################################################################################
 ######################################################################################
 # https://github.com/tensorflow/tensorflow/issues/3270 load two models
 
 
 # hy option2
 # EVALUATE_IMAGES_VAGUE()

#####################################################################################################
##hy: ################                   Test with Webcam                     #######################
#####################################################################################################
if TEST_with_Webcam:
 with tf.Session() as sess:
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
  if ckpt and ckpt.model_checkpoint_path:
   saver.restore(sess, ckpt.model_checkpoint_path)
   print "Evaluation live frames with", ckpt.model_checkpoint_path
  else:
   print 'not found model'
  print 'Test with Webcam starting ...'
  # Camera 0 is the integrated web cam on my netbook
  camera_port = 0
  # EVALUATE_WITH_WEBCAM_track_roi(camera_port)
  EVALUATE_WITH_WEBCAM(camera_port, False)
 
 # cv2.waitKey(0)
 # cv2.destroyAllWindows()
 
 ## TEST with WEBCAM END

cv2.waitKey(0)
cv2.destroyAllWindows()
# hy:total time

#####################################################################################################
##hy: ################                   Test End                             #######################
#####################################################################################################


elapsed_time = time.time() - start_time
print 'Total elapsed time:', "{:.2f}".format(elapsed_time / 60), 'min'

# TODO correcting some of the samples, sometimes the window is a bit large
# TODO Consider bigger images for training, details of a car are not clear in small images
# check at first place if you read images correctly, that incorrecr PIL image that appears at the beginning
# check if 0 is nocar or 1 is nocar
# TODO adding noise can help detection, it can also show power of deep learning as compared to other approaches
# TODO adding noise can show power of deep learning as compared to other approaches
# TODO check above again for making sure
# TODO check print of images for /255 and other numerical compatibility
# TODO check making fake bigger images of the database and see if it works
# TODO check if size of the cars in the street images are appropriate
# TODO try another street image
# TODO adding batch processing ..., researching and reading about batch processing ...
# TODO Histogram normalization or making sure that colors are similar
# TODO change it to correct batch mode, but not Tensorflow batch
# TODO add more negative and better negative examples
# TODO make imbalance between negative and positive samples
# TODO consider confidence measure
# TODO blur images!
# TODO Merge rectangle, use aspect ratio to remove false alarms
# TODO use density of detections in order to remove false alarms
# TODO merge rectangles
# TODO use video cues
# TODO Use a few trained network in parallel, they can only be different in terms of initialization, then vote, it significantly reduces false alarms
# Cars are always correctly detectd, but background detection changes ...
# TODO Save models, with a good name
