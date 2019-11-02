#originally by Hamed, 25Apr.2016
#hy:Changes by Haiyan, change logs are in tensor_train.py
#####################################################################################################
import ImageDraw
import ImageFilter
import ImageOps
import time
from functools import wraps
from random import randint
import os
import sys
import datetime
import settings #hy: collection of global variables
import tools
import prep_image
import cv2
import numpy as np
import tensorflow as tf
from sklearn import datasets
import math
import imutils
from PIL import Image #hy: create video with images
#activate global var
settings.set_global()
start_time = time.time()
#http://lvdmaaten.github.io/tsne/ visualization

do_multiple_test = False #this can only be started with prepared shell script eva.sh
############################################################
if do_multiple_test:
  # use setup in eva.sh
  classifier_model = sys.argv[1] + ".meta"
  n_hidden      = sys.argv[2]
  do_active_fields_test = sys.argv[3]

  print 'do_active', do_active_fields_test
  if do_active_fields_test == '1':
    print 'do_act = 1'
    do_active_fields_test = 1

else:
  # use local setup
  #classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6_6_3conv_L0.72_O1.0_R0.81_U0.75_V1.0_6_0.88-9141.meta"
  #classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6_1_3conv_L0.68_O1.0_R0.85_U0.75_V1.0_1_0.86-7441.meta"
  classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6all_9-871.meta"
  #classifier_model = "..//" + "model_GD360_h184_w184_c6_3conv_L0.7_O1.0_U1.0_7_0.71-6381.meta"
  n_hidden = 360  #n_hidden = classifier_model.split('conv')[1][0:3]
  do_active_fields_test = 0
############################################################

TEST_with_Webcam = False #hy True - test with webcam
video_label = 0 #hy: initialize/default 0:hinten 1:links 2:'oben/', 3:'rechts/', '4: unten/', 5 'vorn/

TEST_with_Images  = True
App_for_Images    = False
TEST_with_Video   = False
video_window_scale = 2

dropout = [0.28, 0.4, 0.5, 0.5]  # 3,4,5,5
#dropout = [0.28]  # 3,4,5,5
dropout_1s = [1] * len(dropout)


#act_min = 0.80
#act_max = 0.93
#add_data = 0 #initial
area_step_size_webcam = 479 #479 #200
TEST_CONV_OUTPUT = False
result_for_table = 1
SAVE_Misclassified     = False
SAVE_CorrectClassified = False

#optimizer_type = 'GD' #'adam' #GD-'gradient.descent'
#learning_rate = 0.0956# TODO 0.05  0.005 better, 0.001 good \0.02, 0.13

#Data
LABEL_LIST = '../FileList.txt'
LABEL_PATH = settings.data_label_path

#LABEL_LIST_TEST = '../FileList_TEST1_sing.txt'

LABEL_LIST_TEST = '../FileList_TEST1.txt'
#LABEL_LIST_TEST = '../FileList_TEST.txt'
LABEL_PATH_TEST = settings.test_label_path

#LABEL_PATH_TEST = "../Test_Images/testpkg_activation/oben/*"

#LABEL_PATH_TEST = "../Test_Images/testpkg2_no_bg/*/*"        #8391 H,L,O,U   42.8 // 7831 H,L,O,U 44.3 //  8421 H,L,O,U 43.2 //
#LABEL_PATH_TEST = "../Test_Images/testpkg3_white_200x200/*/*" #    L,O,  R 43.2           L,O,R 43.2         O,R,V 42.8
#LABEL_PATH_TEST = "../Test_Images/testpkg5big_224x224/*/*"   #    H,O,U  26.7            H,O,U 26.5        H,O,U 0.27

#LABEL_PATH_TEST = "../Test_Images/testpkg6big/*/*" #


GENERATE_FILELIST = 1
log_on = False
DEBUG = 0

# Network Parameters
#n_input = settings.h_resize * settings.w_resize  #hy
n_classes = len(settings.LABELS)

# Active fields test for visualization
if do_active_fields_test == 1:
  print 'To get active fields analysis you must set read_images to sorted read'
  LABEL_PATH_TEST = "../Test_Images/test_active_fields/*/*" #
  LABEL_LIST_TEST = settings.test_label_file_a
  #activation_test_img_name = '../Test_Images/hinten_ori1.jpg' #hinten_t01up  hinten_ori1_rz400
  activation_test_img_name = '../Test_Images/hinten_t01up_1_ex1_22.jpg' #hinten_t01up  hinten_ori1_rz400
  print '-set file', activation_test_img_name


LABELS = settings.LABELS #hy
LABEL_names = settings.LABEL_names #hy

#hy:add timestamp to tensor log files
from datetime import datetime
tensorboard_path = '../Tensorboard_data/sum107/'+str(datetime.now())+'/'


if GENERATE_FILELIST == 1:
  print 'preparing label list'
  tools.prepare_list(LABEL_LIST_TEST, LABEL_PATH_TEST) #hy: avoid wrong list error #hy trial
  print 'loading data'
  tools.read_images(LABEL_LIST_TEST) #hy: get usable input size for w,h
else:
  if TEST_with_Images or TEST_with_Video:
    tools.read_images(LABEL_LIST_TEST)
  else:
    tools.read_images(LABEL_LIST)


class Logger(object):
  def __init__(self):
    self.terminal = sys.stdout
    from datetime import datetime
    str_log = os.path.split(classifier_model)[1][:-5]
    self.log = open(datetime.now().strftime('../logs/log_eva_' + str_log + '.log'), "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    # this flush method is needed for python 3 compatibility.
    # this handles the flush command by doing nothing.
    # you might want to specify some extra behavior here.
    pass


if log_on and (TEST_with_Video):
  sys.stdout = Logger()

############################################################################################################
# hy: initialize crop frame (interest of area in demo window)
# At the moment, this window has to be adjusted to focus our object.
# Different area shown in focus region leads to different test  result.
############################################################################################################
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
  c,r,w,h = 900,650,400,400
  track_window = (c,r,w,h)

  # Create mask and normalized histogram
  roi = frame[r:r+h, c:c+w]
  hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

  mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

  roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

  cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

  term_cond = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1) #hy: TERM_CRITERIA_EPS - terminate iteration condition

  while True:
    ret, frame = video.read()
    if ret:
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
      ret, track_window = cv2.meanShift(dst, track_window, term_cond)

      x,y,w,h = track_window

      #hy: draw rectangle as tracked window area
      cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
      cv2.putText(frame, 'Tracked', (x-25,y-10), cv2.FONT_HERSHEY_SIMPLEX,
          1, (255,255,255), 2, cv2.CV_AA)

      cv2.imshow('Tracking', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
      print 'no frame received'
      break

  return [track_window]


def classify_image(sess, test_image, test_labels, video_frame_i, frame, frame_crop_color,
                   confMat2_TEST_Video, crop_y1, crop_y2, crop_x1, crop_x2, screen_out=True):
 print 'crop_y1, crop_y2, crop_x1, crop_x2', crop_y1, crop_y2, crop_x1, crop_x2
 print 'frame shape', frame.shape[0], frame.shape[1]
 ######################### Tensorflow
 batch_xs, batch_ys = test_image, test_labels

 output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})
 output = tools.convert_to_confidence(output)
 np.set_printoptions(precision=2)

 print '\nFrame', video_frame_i
 tools.print_label_title_conf()

 RES = np.argmax(output)
 label_pred_str = settings.LABELS[RES][:-1]
 # hy: for sub-classes
 # label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
 # RES_sub_to_face = class_label #hy added
 #video_label = 3
 print "label, predict =", video_label, ', ', RES  # hy
 #print "label, predict =", video_label, ', ', RES  # hy

 if RES == video_label:
  label2_TEST_Video = 0
  pred2_TEST_Video = 0

  name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % video_frame_i
  tools.SAVE_CorrectClassified_frame(name_str, frame_crop_color, SAVE_CorrectClassified)  # (v)

 else:
  label2_TEST_Video = 1
  pred2_TEST_Video = 1
  name_str = settings.Misclassified + "/frame_crop%d.jpg" % video_frame_i
  tools.SAVE_Misclassified_frame(name_str, frame_crop_color, SAVE_Misclassified)

 confMat2_TEST_Video[label2_TEST_Video, pred2_TEST_Video] = confMat2_TEST_Video[
                                                             label2_TEST_Video, pred2_TEST_Video] + 1

 cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=2)

 if screen_out:
  prob_str = str(output[0][RES])[:4]  # label index starts from 0
  # prob_str = str(output[0][3])[:4]
  cv2.putText(frame, "predicted: " + label_pred_str + ' acc:' + prob_str, org=(frame.shape[1] / 3, frame.shape[0] / 10),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=4)
  # cv2.putText(frame, label_pred_str, org=(800, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3 )

  # print 'video.get',str(video.get(1))
  # cv2.putText(frame, "prob:" + prob_str, org=(w_frame / 10, h_frame / 8),
  #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
  cv2.putText(frame, 'frame ' + str(video_frame_i), org=(20, 50),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

 frame_demo = imutils.resize(frame, width=600)

 cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
 cv2.imshow("Demo", frame_demo)

 tp = confMat2_TEST_Video[0, 0]
 tn = confMat2_TEST_Video[1, 1]
 # print confMat2_TEST_Video
 # print 'tp, tn, total number of test images:', tp, ', ', tn, ', ', tp + tn
 print confMat2_TEST_Video
 print 'TEST acc:', "{:.4f}".format(tp / (tp + tn))

 cv2.waitKey(20)
 return frame


def do_classification(frame,video_frame_i,frame_crop_color,frame_crop,confMat2_TEST_Video, crop_y1, crop_y2,
                      crop_x1, crop_x2,n_classes,screen_out):
 new_graph = tf.Graph()
 with tf.Session(graph=new_graph) as sess:
  # load classifier model
  #sess, saver = tools.load_classifier_model(sess, '../testbench/6classes/', classifier_model=classifier_model)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../testbench/6classes/")
  saver = tf.train.import_meta_graph(classifier_model)  # (v)

  ckpt.model_checkpoint_path = classifier_model[:-5]

  if ckpt and ckpt.model_checkpoint_path:
   saver = tf.train.Saver()
   saver.restore(sess, ckpt.model_checkpoint_path)
   print "Evaluation with video, model", ckpt.model_checkpoint_path
  else:
   print 'not found model'

  tensorImgIn = frame_crop_color
  #tensorImgIn = cv2.imread('../testbench/frame_color_tmp.jpg')
  cv2.namedWindow('tensorImgIn', cv2.WINDOW_NORMAL)
  cv2.putText(tensorImgIn, 'frame ' + str(video_frame_i), org=(320 / 10, 320 / 8),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
              color=(0, 255, 0), thickness=2)
  cv2.imshow('tensorImgIn', tensorImgIn)

  # resize to fit classifier model
  frame_crop_resize_gray = imutils.resize(cv2.cvtColor(tensorImgIn, cv2.COLOR_BGR2GRAY),
                                          width=settings.w_resize)

  im = frame_crop_resize_gray
  im = np.asarray(im, np.float32)

  test_image = im
  test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

  test_image = im.reshape((-1, im.size))
  test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  # test_image = test_image / 255  #
  test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

  # send to classifier model
  classify_image(sess, test_image, test_labels, video_frame_i, frame, frame_crop_color,
                                  confMat2_TEST_Video, crop_y1, crop_y2, crop_x1, crop_x2,screen_out)


def EVALUATE_IMAGES(session,num_class,img_list,_labels):
  sess = session
  LABEL_LIST_TEST = img_list
  LABELS = _labels
  n_classes = num_class

  ################### active field test part one ################################
  if do_active_fields_test == 1:
    carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
    TEST_length = len(carimages)

    print '1 file',LABEL_LIST_TEST, 'path', LABEL_PATH_TEST, 'len', TEST_length
    #TEST_length = 1
    print 'get active fields'
    row = 0
    col = 0
    print 'filename', activation_test_img_name
    test_img_bg = cv2.imread(activation_test_img_name)#vorn_ww1_1_ex3_4_rz400
    h,w,ch = test_img_bg.shape
    print 'h,w,ch', h,w,ch
    #test_img_bg = cv2.resize(test_img_bg, (400, 400))
    #overlay = np.zeros([400, 400, 3], dtype=np.uint8)
    overlay = np.zeros([h, w, ch], dtype=np.uint8)
    test_img_transparent = overlay.copy()

    cv2.rectangle(overlay, (0, 0), (h, w), color=(60, 80, 30, ch))
    alpha = 0.7  # hy: parameter for degree of transparency
    #cv2.addWeighted(overlay, alpha, test_img_bg, 1 - alpha, 0, test_img_transparent)
    test_img_transparent = tools.add_colorOverlay(overlay,test_img_bg)
    bg = Image.fromarray(test_img_transparent)
  else:
    carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
    TEST_length = len(carimages)
    print 'file', LABEL_LIST_TEST, 'path', LABEL_PATH_TEST

  if DEBUG == 1 and do_active_fields_test == 1:
    overlay_show = Image.fromarray(overlay)
    overlay_show.save('./1-overlay.jpg')
    bg.save('./1-before.jpg')

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
  confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float) #hy collect detailed confusion matrix
  confMat2_TEST = np.zeros((2, 2), dtype=np.float)
  confMat3 = np.zeros((1, n_classes), dtype=np.float)
  count_labels = np.zeros((1, n_classes), dtype=np.float)
  class_probability = np.zeros((1, n_classes), dtype=np.float)
  pred_collect = []
  if result_for_table == 1:
    print 'True/False', 'No.', 'Name', 'TargetLabel', 'PredictLabel', 'Precision','whole_list','Top1','Top1_pres', \
      'Top2', 'Top2_pres','Top3','Top3_pres','Top4','Top4_pres','Top5','Top5_pres','last','last_pres'


  for i in range(0, TEST_length, 1):
    # hy:extra Debug
    # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape

    im = carimages[i]
    # im = frame_crop_resize_gray  # Lazy

    im = np.asarray(im, np.float32)

    CONF = 0.20

    test_image = im

    test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

    # hy: info
    # print "Image size (wxh):", im.size #hy

    # Doing something very stupid here, fix it!
    test_image = im.reshape((-1, im.size))

    #print test_image
    #print sess.run(test_image)

    test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
    test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

    batch_xs, batch_ys = test_image, test_labels

    # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
    #output = sess.run("Accuracy:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
    output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})
    #output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})


    # print("Output for external=",output)
    output = tools.convert_to_confidence(output)  #
    np.set_printoptions(precision=3)

    RES = np.argmax(output) #hy predicted label

    label_target = int(cartargets[i]) #hy ground truth label
    #label_pred_str, label_pred_num = tools.convert_result(RES)
    #label_target_str, label_target_num = tools.convert_result(label_target)

    sorted_vec, prob_all = tools.rank_index(output[0],label_target,result_for_table=result_for_table)
    pred_collect.append(prob_all[0])



    ################### active field test part two begin ################################
    if do_active_fields_test == 1:
      print 'second step, activation test'
      if col >= 4:
        col = 0
        row += 1
      if row >= 4:
        row = 0
      slice_size = int(w/4)# 73 #100
      positions = ((col)*slice_size, (row)*slice_size, (col+1)*slice_size, (row+1)*slice_size) #x0,y0, x1,y1
      print (col)*slice_size, (row)*slice_size, (col+1)*slice_size, (row+1)*slice_size
      col += 1


      #define image for obtaining its active fields

      #activation_test_img = Image.open('../hintenTest.jpg')
      #activation_test_img = Image.open('./vornTest.jpg')
      #activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_u870_400400.jpg')
      #activation_test_img = Image.open('../Test_Images/hinten_ori1_rz400.jpg')
      #activation_test_img = Image.open('../tmp/resized/oben/oben_t2_1_rz400_u856_400400.jpg')
      #activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_u923_400400.jpg')
      #activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_u825_400400.jpg')
      #activation_test_img_copy = cv2.clone(activation_test_img)

      activation_test_img = Image.open(activation_test_img_name)

      thresh = float(max(pred_collect)*0.97)
      print 'thresh', thresh
      if prob_all[0] > thresh:
        #print '\nactive field', positions
        image_crop_part = activation_test_img.crop(positions)
        image_crop_part = image_crop_part.filter(ImageFilter.GaussianBlur(radius=1))
        #bg.paste(image_crop_part, positions)
        np.hstack(image_crop_part, positions)
      print 'save active fields result'
      bg.save('../active_fields_' + os.path.split(classifier_model)[1][:-5] + '.jpg')

    ################### active field test end  ################################

    if result_for_table == 1:
        if LABELS[label_target][:-1] == LABELS[RES][:-1]:
          print '\nTestImage',i+1,f[i],LABELS[label_target][:-1]\
              ,LABELS[RES][:-1],prob_all[0],
          for img_i in xrange(n_classes):
            print settings.LABEL_names[sorted_vec[n_classes-1-img_i]], prob_all[img_i],
        else:
          print '\nMis-C-TestImage',i+1,f[i],LABELS[label_target][:-1],\
              LABELS[RES][:-1],prob_all[0],
          for img_i in xrange(n_classes):
            print settings.LABEL_names[sorted_vec[n_classes-1-img_i]], prob_all[img_i],

    if result_for_table == 0:
      print '\nTestImage', i + 1, ':', f[i]
      # print 'Image name', carimages
      print 'Ground truth label:', LABELS[label_target][:-1], ';  predict:', LABELS[RES][:-1]  # hy
      # print 'Target:', label_target, ';  predict:', RES  # hy

      print '\nRank list of predicted results'
      tools.rank_index(output[0], label_target,result_for_table=result_for_table)

    label = label_target
    predict = int(RES)

    confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1

    count_labels[:, label] = count_labels[:, label] + 1

    if predict == label_target:
      label2_TEST = 0
      pred2_TEST = 0

      confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
      tools.SAVE_CorrectClassified_Img(f[i],SAVE_CorrectClassified)


    else:
      label2_TEST = 1
      pred2_TEST = 1
      tools.SAVE_Misclassified_Img(f[i],SAVE_Misclassified)


    confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
    tp = confMat2_TEST[0, 0]
    tn = confMat2_TEST[1, 1]

  #print summary
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
    wc1 = sess.run("wc1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    wc2 = sess.run("wc2:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    wd1 = sess.run("wd1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    w_out = sess.run("w_out:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    bc1 = sess.run("bc1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    bc2 = sess.run("bc2:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    bd1 = sess.run("bd1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    b_out = sess.run("b_out:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    conv_feature = sess.run("conv2:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    #conv_feature_2D_batch = tools.get_feature_map(conv_feature,f,'conv2') #get defined conv value, not sure for conv2

    #featureImg = sess.run("conv2img:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    summary_op = tf.merge_all_summaries()
    test_res = sess.run(summary_op, feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})


    test_writer.add_summary(test_res, 1)
    #print '2D size',len(conv_feature_2D_batch),'\n',sum(conv_feature_2D_batch[:])
    print 'wc1 shape',wc1.shape, 'wc2:',wc2.shape, 'wd1:',wd1.shape,'w_out:',w_out.shape
    print 'bc1 shape         ',bc1.shape, 'bc2:','       ',bc2.shape, 'bd1:    ',bd1.shape,'b_out:   ',b_out.shape
    print 'pred shape', len(pred_collect)

  else:
    print 'no image got'
  return (confMat1_TEST,count_labels,confMat3,class_probability)

def EVALUATE_IMAGES_mul(session,num_class,img_list,_labels): #(eva)
  sess = session
  LABEL_LIST_TEST = img_list
  LABELS = _labels
  n_classes = num_class

  ################### active field test part one ################################
  if do_active_fields_test == 1:
    carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
    TEST_length = len(carimages)

    print '1 file',LABEL_LIST_TEST, 'path', LABEL_PATH_TEST, 'len', TEST_length
    #TEST_length = 1
    print 'get active fields'
    row = 0
    col = 0
    print 'file', activation_test_img_name
    test_img_bg = cv2.imread(activation_test_img_name)
    h,w,ch = test_img_bg.shape
    print 'h,w,ch',h,w,ch
    #test_img_bg = cv2.resize(test_img_bg, (400, 400))
    #overlay = np.zeros([400, 400, 3], dtype=np.uint8)
    overlay = np.zeros([h, w, ch], dtype=np.uint8)
    test_img_transparent = overlay.copy()

    cv2.rectangle(overlay, (0, 0), (h, w), color=(60, 80, 30, ch))
    alpha = 0.7  # hy: parameter for degree of transparency
    cv2.addWeighted(overlay, alpha, test_img_bg, 1 - alpha, 0, test_img_transparent)
    #test_img_transparent = tools.add_colorOverlay(overlay, test_img_bg)
    bg = Image.fromarray(test_img_transparent)
  else:
    carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST, random_read=False)
    TEST_length = len(carimages)
    print 'file', LABEL_LIST_TEST, 'path', LABEL_PATH_TEST

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
  confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float) #hy collect detailed confusion matrix
  confMat2_TEST = np.zeros((2, 2), dtype=np.float)
  confMat3 = np.zeros((1, n_classes), dtype=np.float)
  count_labels = np.zeros((1, n_classes), dtype=np.float)
  class_probability = np.zeros((1, n_classes), dtype=np.float)
  pred_collect = []
  if result_for_table == 1:
    print 'True/False', 'No.', 'Name', 'TargetLabel', 'PredictLabel', 'Precision','whole_list','Top1','Top1_pres', \
      'Top2', 'Top2_pres','Top3','Top3_pres','Top4','Top4_pres','Top5','Top5_pres','last','last_pres'


  for i in range(0, TEST_length, 1):
    # hy:extra Debug
    # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape

    im = carimages[i]


    ################
    # resize to fit classifier model
    #frame_crop_resize_gray = imutils.resize(cv2.cvtColor(tensorImgIn, cv2.COLOR_BGR2GRAY),
    #                                        width=settings.w_resize)
    frame_crop_resize_gray = imutils.resize(im,width=settings.w_resize)

    frame_crop_resize_gray = cv2.resize(frame_crop_resize_gray, (settings.h_resize, settings.w_resize))
    frame_crop_resize_gray = np.float32(frame_crop_resize_gray.reshape(settings.h_resize, settings.w_resize))

    im = frame_crop_resize_gray
    im = np.asarray(im, np.float32)

    test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

    test_image = im.reshape((-1, im.size))
    test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
    # test_image = test_image / 255  #
    test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

    '''
    # im = frame_crop_resize_gray  # Lazy

    im = np.asarray(im, np.float32)

    CONF = 0.20

    test_image = im

    test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

    # hy: info
    # print "Image size (wxh):", im.size #hy

    # Doing something very stupid here, fix it!
    test_image = im.reshape((-1, im.size))

    #print test_image
    #print sess.run(test_image)

    test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
    test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

    '''


    batch_xs, batch_ys = test_image, test_labels

    # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
    output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})
    #output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": [1,1,1,1]})


    # print("Output for external=",output)
    output = tools.convert_to_confidence(output)  #
    np.set_printoptions(precision=3)

    RES = np.argmax(output) #hy predicted label

    label_target = int(cartargets[i]) #hy ground truth label
    #label_pred_str, label_pred_num = tools.convert_result(RES)
    #label_target_str, label_target_num = tools.convert_result(label_target)

    sorted_vec, prob_all = tools.rank_index(output[0],label_target,result_for_table=result_for_table)
    pred_collect.append(prob_all[0])



    ################### active field test part two begin ################################
    if do_active_fields_test == 1:
      if col >= 4:
        col = 0
        row += 1
      if row >= 4:
        row = 0
      slice_size = int(w/4)#73 #100
      positions = ((col)*slice_size, (row)*slice_size, (col+1)*slice_size, (row+1)*slice_size) #x0,y0, x1,y1
      col += 1


      #define image for obtaining its active fields

      #activation_test_img = Image.open('../hintenTest.jpg')
      #activation_test_img = Image.open('../vornTest.jpg')
      #activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/links/links_t2_1_rz400_u870_400400.jpg')
      #activation_test_img = Image.open('../Test_Images/hinten_ori1_rz400.jpg')
      #activation_test_img = Image.open('../tmp/resized/oben/oben_t2_1_rz400_u856_400400.jpg')
      #activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/unten/unten_t2_1_rz400_u923_400400.jpg')
      #activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_d0_0400_1.jpg')
      #activation_test_img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_u825_400400.jpg')
      #activation_test_img_copy = cv2.clone(activation_test_img)

      activation_test_img = Image.open(activation_test_img_name)

      thresh = float(max(pred_collect)*0.97)
      print 'thresh', thresh
      if prob_all[0] > thresh:
        #print '\nactive field', positions
        image_crop_part = activation_test_img.crop(positions)
        image_crop_part = image_crop_part.filter(ImageFilter.GaussianBlur(radius=1))
        bg.paste(image_crop_part, positions)
        #np.hstack((image_crop_part, positions))
      bg.save('../active_fields_' + os.path.split(classifier_model)[1][:-5] + '.jpg')

    ################### active field test end  ################################

    if result_for_table == 1:
        if LABELS[label_target][:-1] == LABELS[RES][:-1]:
          print '\nTestImage',i+1,f[i],LABELS[label_target][:-1]\
              ,LABELS[RES][:-1],prob_all[0],
          for img_i in xrange(n_classes):
            print settings.LABEL_names[sorted_vec[n_classes-1-img_i]], prob_all[img_i],
        else:
          print '\nMis-C-TestImage',i+1,f[i],LABELS[label_target][:-1],\
              LABELS[RES][:-1],prob_all[0],
          for img_i in xrange(n_classes):
            print settings.LABEL_names[sorted_vec[n_classes-1-img_i]], prob_all[img_i],

    if result_for_table == 0:
      print '\nTestImage', i + 1, ':', f[i]
      # print 'Image name', carimages
      print 'Ground truth label:', LABELS[label_target][:-1], ';  predict:', LABELS[RES][:-1]  # hy
      # print 'Target:', label_target, ';  predict:', RES  # hy

      print '\nRank list of predicted results'
      tools.rank_index(output[0], label_target,result_for_table=result_for_table)

    label = label_target
    predict = int(RES)

    confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1

    count_labels[:, label] = count_labels[:, label] + 1

    if predict == label_target:
      label2_TEST = 0
      pred2_TEST = 0

      confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
      tools.SAVE_CorrectClassified_Img(f[i],SAVE_CorrectClassified)


    else:
      label2_TEST = 1
      pred2_TEST = 1
      tools.SAVE_Misclassified_Img(f[i],SAVE_Misclassified)


    confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
    tp = confMat2_TEST[0, 0]
    tn = confMat2_TEST[1, 1]

  #print summary
  print '\n\nCount correctly classified'
  tools.print_label_title()
  print confMat3

  print 'Total labels'
  print count_labels

  print '\nProportion of correctly classified'
  for pos in range(0, n_classes, 1):
    if count_labels[:, pos] > 0:
      class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
  mAP = round( ( sum(class_probability[0])/ n_classes), 2)
  print class_probability, 'mAP:', mAP

  # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
  # print confMat2_TEST
  print '\nTEST general count:'
  print confMat2_TEST
  overall_acc = round( tp / TEST_length , 2)
  print 'TEST overall acc:', overall_acc


  ###################################################################################
  ## Feature output #################################################################
  ###################################################################################

  if TEST_CONV_OUTPUT:
    print '\nTEST feature output:'

    test_writer = tf.train.SummaryWriter(tensorboard_path + settings.LABELS[label_target], sess.graph)
    wc1 = sess.run("wc1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    wc2 = sess.run("wc2:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    wd1 = sess.run("wd1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    w_out = sess.run("w_out:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    bc1 = sess.run("bc1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    bc2 = sess.run("bc2:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    bd1 = sess.run("bd1:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})
    b_out = sess.run("b_out:0",feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    conv_feature = sess.run("conv2:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    #conv_feature_2D_batch = tools.get_feature_map(conv_feature,f,'conv2') #get defined conv value, not sure for conv2

    #featureImg = sess.run("conv2img:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})

    summary_op = tf.merge_all_summaries()
    test_res = sess.run(summary_op, feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1})


    test_writer.add_summary(test_res, 1)
    #print '2D size',len(conv_feature_2D_batch),'\n',sum(conv_feature_2D_batch[:])
    print 'wc1 shape',wc1.shape, 'wc2:',wc2.shape, 'wd1:',wd1.shape,'w_out:',w_out.shape
    print 'bc1 shape         ',bc1.shape, 'bc2:','       ',bc2.shape, 'bd1:    ',bd1.shape,'b_out:   ',b_out.shape
    print 'pred shape', len(pred_collect)

  else:
    print 'no image got'
  return (confMat1_TEST,count_labels,confMat3,class_probability, overall_acc, mAP)

def EVALUATE_IMAGES_VAGUE(n_classes,img_list):
  LABEL_LIST_TEST = img_list
  # Testing
  cartargets, f = tools.read_test_images(LABEL_LIST_TEST)
  #print 'cartargets label', cartargets
  TEST_length = 20
  #TEST_length = len(cartargets)

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


  patch_size = 42 #227
  for i in range(0, TEST_length, 1):
    # hy:extra Debug
    #im = carimages[i]
    # im = frame_crop_resize_gray  # Lazy


    '''
    #hy: option to use numpy.ndarray, but it cannot use attribute 'crop' of Image (integer) object
    img = cv2.imread(f[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(img, width=patch_size, height=patch_size)
    h_b, w_b = img.shape
    print 'h_b', h_b, ', w_b', w_b
    '''

    print 'processing main test image',f[i]

    #hy: use integer image: Image, resize
    img = Image.open(f[i]).convert('LA') #convert to gray
    h_b, w_b = img.size
    #print 'read test image ok', h_b, ', ', w_b
    img = img.resize((patch_size * 2, patch_size * 2), Image.BICUBIC)  # hy:use bicubic
    #h_b, w_b = img.size
    #print 'h_b', h_b, ', w_b', w_b

    test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
    test_image = img
    test_image_label = cartargets[i]
    # Doing something very stupid here, fix it!
    #test_image = im.reshape((-1, im.size))


    # test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
    #test_image = test_image / 255 - 0.5  # TODO here is tricky, double check with respect to the formats

    slices_rec = prep_image.create_test_slices(test_image,patch_size,test_image_label)
    print 'slices with path received', slices_rec
    slices_len = len(slices_rec)

    out_sum = np.zeros((1, n_classes), dtype=np.float)
    out_box = np.zeros((1, n_classes), dtype=np.float)

    #batch_xs, batch_ys = im, cartargets

    #output_im = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

    for j in range(0, slices_len, 1):
      print '\nprocessing slice', j, slices_rec[j]
      #hy read and resize integer object
      #im_s = Image.open(slices_rec[j]) #numpy.ndarray does not have attribute 'crop'
      #im_s = im_s.resize((settings.h_resize, settings.w_resize), Image.BICUBIC)  # hy:use bicubic, resize func reuqires integer object
      #im_s = im_s.convert('LA') #hy convert to gray

      #hy read and resize continuous number object
      im_s = cv2.imread(slices_rec[j]) #result is not integer
      im_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
      im_s = imutils.resize(im_s, width=settings.h_resize, height=settings.w_resize)

      #hy conver to integer object required for tensor
      im_s = np.asarray(im_s, np.float32)

      CONF = 0.20

      (sorted_vec,outputsub)= EVALUATE_IMAGE_SLICES(im_s,f,i,sess, cartargets)
      print 'slice',j, 'result', sorted_vec
      print 'Image slice', slices_rec[j]
      outbox = outputsub
      out_sum = out_sum + outputsub[0]


    # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
    # print confMat2_TEST
    print '\nTEST general count:'

    print out_sum
    print out_sum/slices_len
    outbox[0] = out_sum/slices_len

    output_im,prob_all = tools.rank_index(outbox[0],test_image_label,result_for_table=result_for_table)
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
      tools.SAVE_CorrectClassified_Img(f[i],SAVE_CorrectClassified)


    else:
      label2_TEST = 1
      pred2_TEST = 1
      tools.SAVE_Misclassified_Img(f[i],SAVE_Misclassified)


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

    print 'Proportion of correctly classified for detailed analysis' #ok
    if count_labels_m[:, pos] > 0:
      for pos in range(0, n_classes, 1):
        class_probability_m[:, pos] = confMat_m3[:, pos] / count_labels_m[:, pos]
      print class_probability_m

    print 'TEST overall acc:', "{:.3f}".format(tp / TEST_length)

def EVALUATE_IMAGE_SLICES(img,f,index,sess, cartargets,num_class): #hy todo change dimension to fit tensorflow
  n_classes = num_class
  confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)
  confMat2_TEST = np.zeros((2, 2), dtype=np.float)
  confMat3 = np.zeros((1, n_classes), dtype=np.float)
  count_labels = np.zeros((1, n_classes), dtype=np.float)
  class_probability = np.zeros((1, n_classes), dtype=np.float)

  img_s = img
  i = index
  test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

  # Doing something very stupid here, fix it!
  test_image = img_s.reshape((-1, img_s.size))


  test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  test_image = test_image / 255 - 0.5  # TODO here is tricky, double check with respect to the formats

  batch_xs1, batch_ys1 = test_image, test_labels

  output = sess.run("pred:0", feed_dict={"x:0": batch_xs1, "y:0": batch_ys1, "keep_prob:0": 1.})

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
  #if label == predict:
  confMat1_TEST[label, predict] = confMat1_TEST[label, predict] + 1

  if int(RES) == label_target:
    label2_TEST = 0
    pred2_TEST = 0
    confMat3[:, int(RES)] = confMat3[:, int(RES)] + 1
    tools.SAVE_CorrectClassified_Img(f[i],SAVE_CorrectClassified)

  else:
    label2_TEST = 1
    pred2_TEST = 1
    tools.SAVE_Misclassified_Img(f[i],SAVE_Misclassified)


  #print 'Count classified'
  #tools.print_label_title()
  #print confMat1_TEST

  confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
  tp = confMat2_TEST[0, 0]
  tn = confMat2_TEST[1, 1]
  print '\nCount correctly classified'
  tools.print_label_title()
  print confMat3

  #print 'Total labels'
  #print count_labels

  #print 'Proportion of correctly classified'
  #if count_labels[:, pos] > 0:
    #for pos in range(0, 6, 1):
    #  class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
    #print class_probability

  #print '\nRank list of predicted results'
  sorted_vec,prob_all = tools.rank_index(output[0], label_target,result_for_table=result_for_table)


  #return (confMat1_TEST, confMat2_TEST, confMat3, count_labels, class_probability,sorted_vec,output)
  return (sorted_vec,output)

def EVALUATE_WITH_WEBCAM(camera_port, stop,num_class): #(cam)
  n_classes = num_class
  #hy: check camera availability
  camera = cv2.VideoCapture(camera_port)

  frame_i = 0
  # hy: initialize confmatrix
  confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)

  #if stop == False:
  #if ckpt and ckpt.model_checkpoint_path:
    # Camera 0 is the integrated web cam

    # Number of frames to throw away while the camera adjusts to light levels
  #  ramp_frames = 1

  while True and not stop: #hy: confirm camera is available
      # Now we can initialize the camera capture object with the cv2.VideoCapture class.
      # All it needs is the index to a camera port.
      print 'Getting image...'

      ret, frame = camera.read()

      # Captures a single image from the camera and returns it in PIL format

      #ret = camera.set(3, 320) #hy use properties 3 and 4 to set frame resolution. 3- w, 4- h
      #ret = camera.set(4, 240)

      cv2.waitKey(1)
      # A nice feature of the imwrite method is that it will automatically choose the
      # correct format based on the file extension you provide.
      # cv2.imwrite(file, camera_capture)

      ####################################  /////////////////////////////


      if frame is not None:
        # print 'frame from webcam obtained'

        # hy: before continue check if image is read correctly
        # while frame is not None:
        frame_i += 1
        # hy:
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]  # hy: h 1536 x w 2304

        # hy: info
        print "h_video and w_video", h_frame, ",", w_frame

        # cv2.imshow("ori", frame)

        #crop_x1 = int((w_frame - area_step_size_webcam) / 2)
        #crop_y1 = int((h_frame - area_step_size_webcam) / 2)  # 1#200

        #crop_x2 = crop_x1 + area_step_size_webcam
        #crop_y2 = int(crop_y1 + area_step_size_webcam * settings.h_resize / settings.w_resize)

        crop_y1 = int((h_frame - area_step_size_webcam) / 2)  # 1#200
        crop_x1 = int((w_frame - area_step_size_webcam) / 2)

        crop_y2 = crop_y1 + area_step_size_webcam #hy:define shorter side as unit length to avoid decimal
        crop_x2 = crop_x1 + area_step_size_webcam * settings.w_resize/settings.h_resize

        #print "x1,y1,x2,y2", crop_x1, 'x', crop_y1, ',', crop_x2, 'x', crop_y2
        # Crop
        # hy: select suitable values for the area of cropped frame,
        #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
        frame_crop_color = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # hy: info
        # print "shape:y1,y2,x1,x2:", crop_y1," ", crop_y2," ", crop_x1," ", crop_x2
        # print "Shape of cropped frame:", frame_crop.shape  #hy: it should be same as the shape of trained images(the input image)


        screen_out = True

        do_classification(frame, frame_i, frame_crop_color, confMat2_TEST_Video, crop_y1, crop_y2,
                          crop_x1, crop_x2, n_classes, screen_out)

        '''
        ##########################
        # Resize
        # frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
        # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
        frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY),
                                                width=settings.w_resize)

        # hy:extra Debug
        # print "Shape of cropped frame in grayscale:", frame_crop_resize_gray.shape

        im = frame_crop_resize_gray  # Lazy

        im = np.asarray(im, np.float32)

        cv2.imshow("TensorFlow Window", imutils.resize(im.astype(np.uint8), 227)) #hy trial

        # Adding noise to the street image #TODO
        # im=add_noise(im,5)

        # Bluring the image to help detection #TODO
        # im = cv2.GaussianBlur(im,(5,5),0)

        CONF = 0.20

        test_image = im

        test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

        # hy: info
        # print "Image size (wxh):", im.size #hy

        # Doing something very stupid here, fix it!
        test_image = im.reshape((-1, im.size))
        # print test_image

        test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)

        # print test_image


        test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

        batch_xs, batch_ys = test_image, test_labels

        # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys

        output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})


        # print("Output for external=",output)
        # print output
        output = tools.convert_to_confidence(output)
        np.set_printoptions(precision=2)
        print '\nFrame', i
        tools.print_label_title_conf()
        print 'confidence =', output  # hy

        RES = np.argmax(output)
        label_pred_str = LABELS[RES][:-1]

        #label_pred_str, label_pred_num = tools.convert_result(RES)
        #print 'label_pred_str', label_pred_str
        print 'predicted label:', LABELS[RES][:-1]

        if label_pred_str == video_label:
          label2_TEST_Video = 0
          pred2_TEST_Video = 0

          name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % i
          tools.SAVE_CorrectClassified_frame(name_str,frame_crop,SAVE_CorrectClassified)


        else:
          label2_TEST_Video = 1
          pred2_TEST_Video = 1

          name_str = settings.Misclassified + "/frame_crop%d.jpg" % i
          tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)


        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)


        #cv2.putText(frame, "predicted1: " + label_pred_str, org=(w_frame / 10, h_frame / 20),
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
        '''
        #TODO add termination condition

      print 'no frame retrieved'

  del(camera)
  return stop


def EVALUATE_WITH_WEBCAM_track_roi(camera_port,num_class):
  n_classes = num_class
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
      print 'xt,yt,wt,ht:', xt, ',', yt, ',', wt, ',' , ht

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
      print "shape after set_video_window:y1, y2, x1, x2:", crop_y1,',', crop_y2, ',', crop_x1, ',',  crop_x2


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

      test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

      # hy: info
      # print "Image size (wxh):", im.size #hy

      # Doing something very stupid here, fix it!
      test_image = im.reshape((-1, im.size))
      # print test_image

      test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)

      # print test_image


      test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

      batch_xs, batch_ys = test_image, test_labels

      # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys

      output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})

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

        name_str = settings.CorrectClassified + "/frame_crop%2d.jpg" % frame_index_i
        tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)


      else:
        label2_TEST_Video = 1
        pred2_TEST_Video = 1

        name_str = settings.Misclassified + "/frame_crop%d.jpg" % frame_index_i
        tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)

      cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=1)

      cv2.putText(frame, "predicted1: " + LABELS[RES], org=(w_frame / 10, h_frame / 20),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

      #cv2.putText(frame, "predicted2: " + LABELS[RES], org=(w_frame / 10, h_frame / 20),
      #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0, 255, 0), thickness=5)

      output_display = str(output[0][RES])[:4]
      cv2.putText(frame, "prob:" + output_display, org=(w_frame / 10, h_frame / 8),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

      #cv2.putText(frame, "predicted1: " + LABELS[RES] + ", prob:" + output[RES], org=(w_frame / 6, h_frame / 10),
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

      #hy TODO add termination condition


  del(camera)

def Evaluate_VIDEO_track_roi(VIDEO_FILE,num_class):
  n_classes = num_class

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

  c,r,w,h = 600,450,600,600

  track_window = (c, r, w, h)
  #track_window = (x0, y0, w, h)

  # Create mask and normalized histogram
  roi = frame[r:r+h, c:c+w]
  hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

  mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

  roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

  cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

  term_cond = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1) #hy: TERM_CRITERIA_EPS - terminate iteration condition

  # hy: initialization of confmatrix
  confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)
  video_frame_i = 0

  while True:

    ret, frame = video.read()
    if ret:
      video_frame_i = video_frame_i + 1

      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
      ret, track_window = cv2.meanShift(dst, track_window, term_cond)

      xt, yt, wt, ht = track_window
      #hy info
      #print 'xt,yt,wt,ht:', xt, ',', yt, ',', wt, ',' , ht

      #hy: draw rectangle as tracked window area
      cv2.rectangle(frame, (xt,yt), (xt+wt,yt+ht), 255, 2)
      cv2.putText(frame, 'tracked', (xt-25,yt-10), cv2.FONT_HERSHEY_SIMPLEX,
          1, (255,255,255), 2, cv2.CV_AA)

      cv2.waitKey(500)

      ###############################################################################################
      #   evaluate
      ###############################################################################################
      # hy: info
      #print "shape in evaluate:x1,y1:", crop_x1, ',', crop_y1

      crop_x1 = xt
      crop_x2 = xt + wt
      crop_y1 = yt
      area_step_size = ht
      crop_y2 = crop_y1 + area_step_size * settings.h_resize / settings.w_resize


      # hy: select suitable values for the area of cropped frame,
      #    adapt to the ratio of h to w after resized, e.g. 42x42 ie.w=h
      frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

      # hy: info
      #print "shape after set_video_window:y1, y2, x1, x2:", crop_y1,',', crop_y2, ',', crop_x1, ',',  crop_x2


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

      test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

      # hy: info
      # print "Image size (wxh):", im.size #hy

      # Doing something very stupid here, fix it!
      test_image = im.reshape((-1, im.size))
      # print test_image

      test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)

      # print test_image


      test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

      batch_xs, batch_ys = test_image, test_labels

      #print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys

      output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})

      # print("Output for external=",output)
      # print output
      output = tools.convert_to_confidence(output)
      np.set_printoptions(precision=2)

      print '\nFrame', video_frame_i
      tools.print_label_title_conf()
      print 'confidence =', output  # hy

      RES = np.argmax(output)

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

      #cv2.imshow('Tracking', frame)
      #if cv2.waitKey(1) & 0xFF == ord('q'):
       #   break
    #else:
      #print 'no frame received for tracking'
      #break

def EVALUATE_VIDEO(VIDEO_FILE,num_class): #(v)
  n_classes = num_class
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

      # print "h_video and w_video", h_resize, ",", w_resize

      # hy: select suitable values for the area of cropped frame,
      use_focus_window = 0
      if use_focus_window:
        #crop_x1 = 550  # 550
        #crop_y1 = 700  # 700# 300
        #area_step_size = 740  # 740# 640
        crop_x2 = crop_x1 + area_step_size * 1
        crop_y2 = crop_y1 + area_step_size * 1 * settings.h_resize / settings.w_resize
        frame_crop_color = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        # cv2.imwrite('../testbench/frame_color_tmp.jpg', np.uint8(frame_crop_color))
      else:
        crop_x1 = 0
        crop_y1 = 0

        crop_x2 = 0 + w_frame  # 2300  #1920
        crop_y2 = 0 + h_frame  # 1536  #1080
        frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        frame_crop_color = frame_crop.copy()

      # hy: info
      print "shape:y1,y2,x1,x2:", crop_y1,", ", crop_y2,", ", crop_x1,", ", crop_x2


      # Resize
      #frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=settings.w_resize)
      # hy: it is something different to video_crop_tool.py, here for tensorflow the width has to be that of input image
      #cv2.imshow('frame',frame)

      #tools.SAVE_CorrectClassified_frame('../classified/frame_' + str(video_frame_i) + '.jpg', frame, save=False)

      screen_out = True

      do_classification(frame, video_frame_i, frame_crop_color, confMat2_TEST_Video, crop_y1, crop_y2,
                        crop_x1, crop_x2, n_classes, screen_out)

      '''
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
      test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

      # hy: info
      # print "Image size (wxh):", im.size #hy

      # Doing something very stupid here, fix it!
      test_image = im.reshape((-1, im.size))
      test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
      # print test_image
      test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
      
      print 'test image shape', test_image.shape
      
      ######################### Tensorflow
      batch_xs, batch_ys = test_image, test_labels

      output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})
      # print("Output for external=",output)
      # print output
      output = tools.convert_to_confidence(output)
      np.set_printoptions(precision=2)

      print '\nFrame', video_frame_i
      tools.print_label_title_conf()

      RES = np.argmax(output)
      label_pred_str = LABELS[RES][:-1]
      #hy: for sub-classes
      #label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
      #RES_sub_to_face = class_label #hy added

      print "label, predict =", video_label, ', ', RES # hy

      if RES == video_label:
        label2_TEST_Video = 0
        pred2_TEST_Video = 0

        name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % video_frame_i
        tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified) #(v)

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
      #cv2.putText(frame, label_pred_str, org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
      #            color=(0, 255, 0), thickness=1)
      frame_demo = imutils.resize(frame, width=1200)

      # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
      cv2.imshow("Demo", frame_demo)
      cv2.waitKey(300)
      '''


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

#def dense_to_one_hot(labels_dense, num_classes=n_classes):
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  if DEBUG == 1:
    print 'one_hot_vector:', labels_one_hot[0]
  return labels_one_hot


#####################################################################################################
##################                 TEST with Video (v)                       ###########################
#####################################################################################################

if TEST_with_Video:
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../testbench/6classes/")
    saver = tf.train.import_meta_graph(classifier_model)  # (v)

    ckpt.model_checkpoint_path = classifier_model[:-5]

    if ckpt and ckpt.model_checkpoint_path:
      saver = tf.train.Saver()
      saver.restore(sess, ckpt.model_checkpoint_path)
      print "Evaluation with video, model", ckpt.model_checkpoint_path
    else:
      print 'not found model'

    print 'Test with video starting ...'
    #for video_index in xrange(1):
    test_video_custom = 1
    if test_video_custom == 1:
      video_list = ['full/']
      TestFace = 'full'
      video_window_scale = 1
      VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
      print VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label
      EVALUATE_VIDEO(VIDEO_FILE, n_classes)
    else:
      video_list = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']

      for video_index in xrange(len(video_list)):
        TestFace = video_list[video_index][:-1] # all # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
        print 'Test face:', TestFace

        #TestFace = settings.LABELS[video_index][:-1] #'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
        VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
        # hy: another option - automatically move ROI downwards, then to the right
        # crop_y1 = crop_y1 + area_step_size/50
        # if crop_y2+area_step_size >= frame.shape[0]:
        # crop_y1 = 0
        # crop_x1 = crop_x1 + 200
        # if crop_x2+area_step_size >= frame.shape[1]:
        ##crop_x1 = 0
        # break


        # hy: info
        #print "shape after set_video_window:x1,y1:", crop_x1, ", ", crop_y1

        #track_frame = track_roi(VIDEO_FILE)
        #Evaluate_VIDEO_track_roi(VIDEO_FILE)

        EVALUATE_VIDEO(VIDEO_FILE, n_classes)
        print 'test face:', TestFace, 'done\n'


#####################################################################################################
##hy: ################                   TEST with IMAGES                     #######################
#####################################################################################################
init = tf.initialize_all_variables() #hy


if TEST_with_Images:
  #model 1
  with tf.Session() as sess:
    #hy: load saved model with values
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../testbench/6classes/") # "../backupModel/"
    saver = tf.train.import_meta_graph(classifier_model) #(eva)
    #saver = tf.train.import_meta_graph("../tensor_model_sum/model_GD3conv360_h184_w184_c63conv_O0.75_U1.0_1_0.79-431.meta") #(eva)

    #model_GD3conv360_h184_w184_c63conv_H0.7_0_0.57_U1.0_0_0.57-901   0.33
    #last 0.26
    #model_GD3conv360_h184_w184_c63conv_U1.0_2_0.65-1291  0.263
    #model_GD3conv360_h184_w184_c63conv_H0.75_2_0.58_U1.0_2_0.58-1131 0.36
    #model_GD3conv200_h184_w184_c6_3conv_O0.73_0_0.42-2961  0.592 (new test files) -- top
    #model_GD2000_h184_w184_c6_II_O0.64_0_0.42-1061  0.51(new test files)
    #
    ckpt.model_checkpoint_path = classifier_model[:-5]
    
    #sess, saver = tools.load_classifier_model(sess, '../logs/', classifier_model=classifier_model)
    
    if ckpt and ckpt.model_checkpoint_path:
      print "Evaluation with images, model", ckpt.model_checkpoint_path
      saver = tf.train.Saver()
      saver.restore(sess,ckpt.model_checkpoint_path)
    else:
      print 'not found model'
    print 'I-Test with Images starting ...'  #print 'Test with images starting ...', ckpt.model_checkpoint_path

    #sess.run(init)
    #hy: evaluate
    eval_file_list = LABEL_LIST_TEST
    #LABELS = ['links/', 'rechts/']
    confMat1_TEST_i,count_labels,confMat3,class_probability,overall_acc,mAP = EVALUATE_IMAGES_mul(sess,6, eval_file_list, LABELS) #
    '''
    print '\nCount correctly classified'
    tools.print_label_title()
    print confMat3

    print 'Total labels'
    print count_labels

    print '\nProportion of correctly classified'
    for pos in range(0, n_classes, 1):
      if count_labels[:, pos] > 0:
        class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
        print class_probability
    '''
    print 'Count classified in each class for detailed analysis'
    tools.print_label_title()
    print confMat1_TEST_i

    if do_active_fields_test == 0 and do_multiple_test:
      c_short = os.path.basename(classifier_model)[:-5]
      for filename in os.dirlist('../logs/'):
        if filename.startwith('eva-' + classifier_model + '.txt'):
          os.rename('eva-' + c_short + '.txt', 'eva-' + c_short + '-' + str(overall_acc) + '-' + str(mAP) + '.txt')

    #  print '\n[Hint] If error, check settings - tensor input size and n_class, n_hidden all should be the same as given in log file name'
    #  print '\n[Hint-e]', str(e)

  #model 2
  '''
  new_graph = tf.Graph()
  with tf.Session(graph=new_graph) as sess2:
    # method 2 must initial sess
    # hy: load saved model with values
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../tensor_model_pre/")  # ../backupModel/
    saver2 = tf.train.import_meta_graph("../tensor_model_pre/model_GD60_h42_w42_c6_II1821.meta")
    if ckpt and ckpt.model_checkpoint_path:
      print '\n################################\nsecond model'
      saver2 = tf.train.Saver()
      saver2.restore(sess2, ckpt.model_checkpoint_path)
    else:
      print 'not found model'
    print 'II-Test with Images starting ...'  # print 'Test with images starting ...', ckpt.model_checkpoint_path

    #sess.run(init)
    # hy: evaluate
    eval_file_list2 = LABEL_LIST_TEST
    LABELS = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
    confMat1_TEST_i = EVALUATE_IMAGES(sess2,6,eval_file_list2, LABELS)
    print 'Count classified in each class for detailed analysis'
    tools.print_label_title()
    print confMat1_TEST_i
    '''
    #####################################################################################################
    ##hy: ################                   App for IMAGES                     #######################
    #####################################################################################################
init = tf.initialize_all_variables()  # hy

def get_precision(session,im):
  sess = session
  im = np.asarray(im, np.float32)

  CONF = 0.20

  test_image = im

  test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict

  # Doing something very stupid here, fix it!
  test_image = im.reshape((-1, im.size))

  # print test_image
  # print sess.run(test_image)

  test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
  test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats

  # hy: evaluate
  batch_xs, batch_ys = test_image, test_labels

  # output = sess.run("Accuracy:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
  output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})

  # print("Output for external=",output)
  output = tools.convert_to_confidence(output)  #

  return output

if App_for_Images:
  # model 1
  # '''
  #total_images, digits, carimages, cartargets, f, val2_digits, val2_images, val2_targets, val2_f = import_data()
  carimages, cartargets, f = tools.read_images(LABEL_LIST_TEST)
  print '1 file', LABEL_LIST_TEST
  TEST_length = len(carimages)
  # carimages = carimages / 255 - 0.5  #TODO here is tricky, double check wit respect to the formats
  confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
  confMat2_TEST = np.zeros((2, 2), dtype=np.float)
  confMat3 = np.zeros((1, n_classes), dtype=np.float)
  count_labels = np.zeros((1, n_classes), dtype=np.float)
  class_probability = np.zeros((1, n_classes), dtype=np.float)

  for i in range(0, TEST_length, 1):
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
      # hy: load saved model with values
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../backupModel/")  # ../backupModel/
      saver = tf.train.import_meta_graph("model_graph_GD_c6_491_oben_unten.meta")
      if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print 'not found model'
      im = carimages[i]
      # im = frame_crop_resize_gray  # Lazy
      output1 = get_precision(sess,im)
      tools.print_label_title()
      np.set_printoptions(precision=3)
      print 'output1', output1
    # model 2
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess2:
      # method 2 must initial sess after adding operation and before run
      # hy: load saved model with values
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")  # ../backupModel/
      saver2 = tf.train.import_meta_graph("model_graph_GD_c6_601_links_25_oben_91_unten_83_vorn_70.meta")
      if ckpt and ckpt.model_checkpoint_path:
        print '################################\nsecond model'
        saver2 = tf.train.Saver()
        saver2.restore(sess2, ckpt.model_checkpoint_path)
      else:
        print 'not found model'
      print 'II-Test with Images starting ...'  # print 'Test with images starting ...', ckpt.model_checkpoint_path

      #sess2.run(init)
      # hy: evaluate
      output2 = get_precision(sess2,im)
      tools.print_label_title()
      print 'output2', output2
      output = output1+output2
      print 'output', output
      RES = np.argmax(output)  # hy predicted label

      label_target = int(cartargets[i])  # hy ground truth label
      #print 'label_target',i,':', label_target
      # label_pred_str, label_pred_num = tools.convert_result(RES)
      # label_target_str, label_target_num = tools.convert_result(label_target)

      print '\nTestImage', i + 1, ':', f[i]
      # print 'Image name', carimages
      print 'Ground truth label:', LABELS[label_target][:-1], ';  predict:', LABELS[RES][:-1]  # hy
      # print 'Target:', label_target, ';  predict:', RES  # hy
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

      print '\nRank list of predicted results'
      tools.rank_index(output[0], label_target,result_for_table=result_for_table)

      print '\nCount correctly classified'
      tools.print_label_title()
      print confMat3

      print 'Total labels'
      print count_labels

      print 'Proportion of correctly classified'
      if count_labels[:, pos] > 0:
        for pos in range(0, n_classes, 1):
          class_probability[:, pos] = confMat3[:, pos] / count_labels[:, pos]
        print class_probability

      # print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
      # print confMat2_TEST
      print '\nTEST general count:'
      print confMat2_TEST
      print 'TEST overall acc:', "{:.3f}".format(tp / TEST_length)

    #matrix count
    tools.print_label_title()
    print confMat1_TEST
    print '\n'




    ######################################################################################
    ######################################################################################
    #https://github.com/tensorflow/tensorflow/issues/3270 load two models

  # hy option2
  #EVALUATE_IMAGES_VAGUE()


#####################################################################################################
##hy: ################                   Test with Webcam                     #######################
#####################################################################################################
if TEST_with_Webcam:
  with tf.Session() as sess:
    #saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../testbench/6classes/")
    saver = tf.train.import_meta_graph(classifier_model) #(web)

    ckpt.model_checkpoint_path = classifier_model[:-5]
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print "Evaluation live frames with", ckpt.model_checkpoint_path
    else:
      print 'not found model'
    print 'Test with Webcam starting ...'
    # Camera 0 is the integrated web cam on my netbook
    camera_port = 0
    #EVALUATE_WITH_WEBCAM_track_roi(camera_port,n_classes)
    EVALUATE_WITH_WEBCAM(camera_port, False,n_classes)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

  ## TEST with WEBCAM END



cv2.waitKey(0)
cv2.destroyAllWindows()
# hy:total time

#####################################################################################################
##hy: ################                   Test End                             #######################
#####################################################################################################


elapsed_time = time.time() - start_time
print 'Total elapsed time:', "{:.2f}".format(elapsed_time / 60), 'min'



if __name__ == "__main__":
    pass
