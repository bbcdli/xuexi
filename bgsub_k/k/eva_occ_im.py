import tensorflow as tf
# from import Image
import cv2
import numpy as np
import os
import sys
import time
from keras.backend import set_image_dim_ordering
from keras.models import load_model
import keras
from random import randint
import datetime
import settings  # hy: collection of global variables
import tools
import math
import imutils
from PIL import Image
import re
# from background_learning_s import dice_coef_loss
# tf.python.control_flow_ops = tf #hy:for remote
# KERAS_BACKEND=tensorflow python -c "from keras import backend"
# Using TensorFlow backend.

#####################################################################################################

do_active_fields_test = 0

dropout = [0.15, 0.25, 0.4, 0.5, 1, 0.4, 0.25, 0.15, 0.149]  # 1st
dropout_1s = [1] * len(dropout)
##########

SAVE_Misclassified = 0
SAVE_CorrectClassified = 0
######################################################################################################

# Load 2cl
def get_read_path_and_files_for_im_m_2cl(read_from_file=False):  # represent
  # folders = settings.LABELS
  # folders = ['/']
  total_files_im, total_files_m = [], []
  all_read_path_im, all_read_path_m = [], []
  folders = ['empty/', 'non_empty/']  # Paper order
  # folders = ['training/1A/'] #
  # folders = ['2_bad_recognized/']
  for folder in folders:
    read_path_im = os.path.join(PROJ_DIR, 'Test_Data/1eva_cl/', folder, 'im/')
    read_path_m = os.path.join(PROJ_DIR, 'Test_Data/1eva_cl/', folder, 'm/')
    # read_path_im = PROJ_DIR + 'Data/' + folder + 'before_training/1_images_from_Christoph_folder_out1/1_m_small/'
    # read_path_im = PROJ_DIR + 'Data/' + folder + '1_im_small/'
    # read_path_m = PROJ_DIR + 'Data/' + folder + '1_m/'
    # read_path_m = read_path_im

    files_im = sorted([s for s in os.listdir(read_path_im)])  # if 'offi' in s])
    files_m = sorted([s for s in os.listdir(read_path_m)])  # if 'offi' in s])

    total_files_im = total_files_im + files_im
    total_files_m = total_files_m + files_m

    for i in xrange(len(files_im)):
      all_read_path_im.append(read_path_im)
      all_read_path_m.append(read_path_m)

  if INFO_0:
    print 'num of files:', len(files_im), ',  files[0]:', files_im[0]

  return all_read_path_im, all_read_path_m, total_files_im, total_files_m


def ROI(pred_thresh, im_crop, w, h, im_i=0, save_file=False):
  """
  Finalize prediced ROI
  
  :param pred_thresh: image region to analyze -
  
  :param im_crop: RGB input
  
  :param w,h: width and height to be resized to
  
  :param im_i: serial ID of the input image
  
  :param save_file: switch for saving file
  
  Available options:
  
  save_file = True/ False
  
  """

  # input

  fr = im_crop.copy()
  im_crop_rz = cv2.resize(im_crop, (h, w))
  blackbg = np.zeros((w, h, 3), np.uint8)
  whitebg = np.zeros((w, h, 3), np.uint8)
  whitebg.fill(255)
  new_mask = np.zeros((w, h, 3), np.uint8)

  ###############

  def find_contour(obj_area, thresh):
    gray = cv2.resize(np.uint8(obj_area), (h, w))  # direct
    ret, gray = cv2.threshold(gray, thresh, 255, 0)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 1

    if len(contours) > 0:
      screen_out = True
    else:
      screen_out = False
    return contours, screen_out

  #########################################################

  contours, screen_out = find_contour(pred_thresh, thresh_res)  # 160/255 = 0.62

  if screen_out:
    # debug
    # time for contour 0.000312089920044
    fr_add_cont = cv2.resize(np.uint8(fr), (h, w))  # 0,1
    fr_ori_int8 = fr_add_cont.copy()  # 0,1
    # initialize
    roi_res = im_crop_rz.copy()
    new_roi_res = im_crop_rz.copy()
    old_roi_res = im_crop_rz.copy()
    ######################################
    largest_areas = sorted(contours, key=cv2.contourArea)  # 1
    for i in xrange(len(largest_areas)):
      cv2.drawContours(fr_add_cont, [largest_areas[-i]], 0, (255, 255, 255, 255), -1)
      cv2.drawContours(blackbg, [largest_areas[-i]], 0, (255, 255, 255, 255), -1)
    new_mask = blackbg
    x, y, bw, bh = 0, 0, 0, 0
    r1, r2, c1, c2 = y, y + bh, x, x + bw

  else:
    print 'no contour found (1)'
    screen_out = False
    old_roi_res = np.zeros((w, h, 3), np.uint8)
    old_roi_res.fill(255)
    new_roi_res = old_roi_res.copy()  # no roi
    new_mask = np.zeros((w, h, 3), np.uint8)  # full black mask
    fr_add_cont = im_crop_rz.copy()
    x, y, bw, bh = 0, 0, 0, 0
    r1, r2, c1, c2 = y, y + bh, x, x + bw

  # time for contour,add mask 0.00493407249451

  #####################################################################
  old_mask = blackbg

  return fr_add_cont, old_mask, old_roi_res, new_mask, new_roi_res, r1, r2, c1, c2, screen_out, whitebg


# display
def get_classify_result(sess, test_image, test_labels, im_i, frame, frame_crop,
                        crop_y1, crop_y2, crop_x1, crop_x2, border, screen_out=False, fname='', target=0):
  # print 'frame shape', frame.shape[0], frame.shape[1]
  ######################### Tensorflow
  if not result_for_table:
    print '\nNo.', im_i + 1, ' file:', fname

  # test_image= test_image[crop_y1:crop_y2,crop_x1:crop_x2]
  batch_xs, batch_ys = test_image, test_labels
  crop_y1, crop_y2, crop_x1, crop_x2 = int(frame_crop.shape[0] * crop_y1 / 320), int(
    frame_crop.shape[0] * crop_y2 / 320), int(frame_crop.shape[1] * crop_x1 / 320), int(frame_crop.shape[1] * crop_x2 / 320)


  # print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
  dropout_cl = [0, 0, 0, 0]
  dropout_1s_cl = [1] * len(dropout_cl)
  output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s_cl})
  # print("Output for external=",output)
  output = tools.convert_to_confidence(output)  #
  np.set_printoptions(precision=3)
  
  if DEBUG:
    print 'output', output
  
  rank_outputs = sorted(range(len(settings.LABELS)), key=output[0].__getitem__)
  RES = np.argmax(output)  # hy index starting from 0, 0-5 corresponding predicted label 1-6
  label_pred_str = settings.LABELS_en[RES][:-1]
  
  if target == RES:
    # print 'RES:', RES, 'target', target
    tp_color = (0, 0, 255)
  else:
    tp_color = (0, 0, 0)
  label_pred_str2 = settings.LABELS_en[rank_outputs[-2]][:-1]
  label_pred_str3 = settings.LABELS_en[rank_outputs[-3]][:-1]
  label_pred_str4 = settings.LABELS_en[rank_outputs[-4]][:-1]
  label_pred_str5 = settings.LABELS_en[rank_outputs[-5]][:-1]
  label_pred_str6 = settings.LABELS_en[rank_outputs[-6]][:-1]
  prob_str = str('{:.4f}'.format(output[0][RES]))
  prob_str2 = str('{:.4f}'.format(output[0][rank_outputs[-2]]))
  prob_str3 = str('{:.4f}'.format(output[0][rank_outputs[-3]]))
  prob_str4 = str('{:.4f}'.format(output[0][rank_outputs[-4]]))
  prob_str5 = str('{:.4f}'.format(output[0][rank_outputs[-5]]))
  prob_str6 = str('{:.4f}'.format(output[0][rank_outputs[-6]]))

  if INFO_0:
    print prob_str2, prob_str3, prob_str4, prob_str5, prob_str6

  # hy: for sub-classes
  # label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
  # RES_sub_to_face = class_label #hy added
  # print "target, predict =", target, ', ', RES  # hy
  frame = frame_crop
  if DEBUG:
    print 'frame shape(classify demo):', frame.shape
  
  if frame.shape[0] > 330:
    frontsize, frontsize_no, frontsize_stat, thickness, thickness_no = 2, 1, 1.5, 4, 2
  else:
    frontsize, frontsize_no, frontsize_stat, thickness, thickness_no = 0.4, 0.5, 1.5, 1, 1
  
  if do_classification:  ##---
    print 'No.', im_i + 1, fname, target, label_pred_str, prob_str, label_pred_str2, prob_str2, label_pred_str3, \
   \
      prob_str3, label_pred_str4, prob_str4, label_pred_str5, prob_str5, label_pred_str6, prob_str6
  
  demo_window_width = 300  # 600 * 1080 / 1920
  add_statistic_page = True
  stat = np.zeros((demo_window_width, demo_window_width, 3), np.uint8)
  stat.fill(255)
  txt_col1 = 1
  txt_col2 = 250

  if IMAGE_SIZE == 160:
    rows = [60, 120, 170, 220, 270]
  else:
    rows = [60, 120, 150, 180, 210]  # need to find other appropriate values

  if screen_out and not CLOSE_ALL:
    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=thickness_no)
  
    # frame_demo = imutils.resize(frame, width=600)
    cv2.putText(frame, "predicted top 1: " + label_pred_str + ' confid.:' + prob_str,
                org=(frame.shape[1] / 5, int(frame.shape[0] * 0.1)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(0, 255, 0), thickness=thickness)
  
    if add_statistic_page:  # int(demo_window_width * 0.1)
      cv2.putText(stat, "1: " + label_pred_str,
                  org=(txt_col1, rows[0]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=tp_color,
                  thickness=thickness)
  
      cv2.putText(stat, prob_str,
                  org=(txt_col2, rows[0]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)

  ###########################################################################################################

    cv2.putText(frame, "predicted top 2: " + label_pred_str2 + ' confid.:' + prob_str2,
                org=(frame.shape[1] / 5, int(frame.shape[0] * 0.20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
  
    if add_statistic_page:
      cv2.putText(stat, "2: " + label_pred_str2,
                  org=(txt_col1, rows[1]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
      cv2.putText(stat, prob_str2,
                  org=(txt_col2, rows[1]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
  ###########################################################################################################
    cv2.putText(frame, "predicted top 3: " + label_pred_str3 + ' confid.:' + prob_str3,
                org=(frame.shape[1] / 5, int(frame.shape[0] * 0.25)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
  
    if add_statistic_page:
      cv2.putText(stat, "3: " + label_pred_str3,
                  org=(txt_col1, rows[2]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
      cv2.putText(stat, prob_str3,
                  org=(txt_col2, rows[2]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
  ###########################################################################################################
    cv2.putText(frame, "predicted top 4: " + label_pred_str4 + ' confid.:' + prob_str4,
                org=(frame.shape[1] / 5, int(frame.shape[0] * 0.30)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)

    if add_statistic_page:
      cv2.putText(stat, "4: " + label_pred_str4,
                  org=(txt_col1, rows[3]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
      cv2.putText(stat, prob_str4,
                  org=(txt_col2, rows[3]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)

  ###########################################################################################################
    cv2.putText(frame, "predicted top 5: " + label_pred_str5 + ' confid.:' + prob_str5,
                org=(frame.shape[1] / 5, int(frame.shape[0] * 0.35)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
  
    if add_statistic_page:
      cv2.putText(stat, "5: " + label_pred_str5,
                  org=(txt_col1, rows[4]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
      cv2.putText(stat, prob_str5,
                  org=(txt_col2, rows[4]),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                  thickness=thickness)
  
  ###########################################################################################################
  
    Print_ALL = False
  
    if Print_ALL:
      cv2.putText(frame, "predicted top 6: " + label_pred_str6 + ' confid.:' + prob_str6,
                  org=(frame.shape[1] / 5, int(frame.shape[0] * 0.4)),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
  
      if add_statistic_page:
        cv2.putText(stat, "6: " + label_pred_str6,
                    org=(txt_col1, 240),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                    thickness=thickness)
  
        cv2.putText(stat, prob_str6,
                    org=(txt_col2, 240),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0),
                    thickness=thickness)

    # print 'video.get',str(video.get(1))
  
    frame_demo = imutils.resize(frame,
                                width=demo_window_width)  # cannot use frame because it should be reserved for receiving next input
  
    demo_window_height = int(demo_window_width * frame.shape[0] / frame.shape[1])
    cv2.imshow('stat', stat)
  
    if DEBUG:
      print 'demo height', demo_window_height
  
    cv2.putText(frame_demo, 'No. ' + str(im_i + 1), org=(10, demo_window_height - 16),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_no, color=(0, 255, 0),
                thickness=thickness_no)
  
    if INFO_0:
      cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
      cv2.imshow("Demo", frame_demo)
  


  else:
    frame_demo = imutils.resize(frame, width=demo_window_width)
  
  return int(RES), frame_demo, stat
  

def do_statistics(confMat1, confMat2, data_length):
  # for i in range(0, len(settings.LABELS)):
  #	confMat1[i, :] = confMat1[i, :] / np.sum(confMat1[i, :])

  if not result_for_table:
    tools.print_label_title()

    print confMat1

  tp = confMat2[0, 0]
  tn = confMat2[1, 1]

  overall_acc = round(tp / data_length, 2)

  if not result_for_table:
    print 'TEST overall acc:', overall_acc

  return overall_acc


def do_segment_im(model, im_crop_color, im_crop, im_i, h, w, in_ch, show=False, save=False):
  if INFO_0:
    print w, h

  if in_ch == 1:
    im_crop = imutils.resize(cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY), width=w)
    im_crop = cv2.resize(im_crop, (h, w))
    im_crop = np.float32(im_crop.reshape(h, w)) / 255.0

    image_k_tensor = np.zeros((3, 1, h, w))  # 3,1,320,320 in theano ordering
    image_k_tensor[1, :, :, :] = im_crop
    image_k_tensor = image_k_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape

    image_view = cv2.resize(im_crop_color, (h, w))

    if INFO_0:
      print 'image_tensor 0:', image_k_tensor.shape  # 'th': (ch, h, w),  'tf': (h, w, ch)

  # image_k_tensor = np.transpose(image_k_tensor, (0, 2, 1, 3))
  # print 'image_tensor 1:', image_k_tensor.shape

  else:

    # im_crop = imutils.resize(cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY), width=w)
    im_crop = cv2.resize(im_crop_color, (h, w)) / 255.0
    im_crop = np.rollaxis(im_crop, axis=2, start=0)  # change order of dimenstions

    image_k_tensor = np.zeros((3, 3, h, w))  #
    image_k_tensor[1, :, :, :] = im_crop
    image_k_tensor = image_k_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape

    image_view = cv2.resize(im_crop_color, (h, w))

  # debug

  # print '2-shape of test images', images.shape  # (1, 1, 320, 320)

  ######################

  image_k_tensor = tools.reduce_mean_stdev(image_k_tensor, print_val=False)

  res_pass_list, res_fail_list = [], []
  # print 'images.shape[0]:', images.shape[0] #1

  for i in range(0, image_k_tensor.shape[0]):
    start = time.time()
    pred = model.predict(image_k_tensor[i, :, :, :].reshape(1, in_ch, h, w), batch_size=1)  # MA_im
    end = time.time()
    print 'time elapsed for model.predict:', (end - start), 's'  # 1.24031400681 s

    if DEBUG:
      print 'model direct output,reshape:', pred

    if INFO_0:
      print 'Test image', im_i, ", Min,Max: %f %f" % (np.min(pred), np.max(pred))

    if (np.min(pred) == np.max(pred)) and (np.min(pred) == 0):
      print 'no feature captured'
      break

    # debug

    if DEBUG:
      print 'pred shape', (pred.shape)

    pred = pred[0, 0, :, :].reshape((h, w))  # pred[0,0,:,:]
    pred_255 = pred * 255  #

    if DEBUG:
      print 'model output255,reshape:', pred_255

    pred_int = np.uint8(pred_255)
    pred_thresh = pred_int.copy()

    ceiling_to_pixel_value = 255

    idx = pred_int[:, :] > thresh_res
    pred_thresh[idx] = ceiling_to_pixel_value

    # debug

    # cv2.imwrite('../testbench/frame_res_tmp.jpg', np.uint8(res))

    if INFO_0:
      print '# show segment result for frame', im_i

  #############################################################################

  # IMAGE

  fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, \
 \
  r1, r2, c1, c2, screen_out, roi_whitebg = ROI(pred_thresh, im_crop_color, h, w, im_i, save_file=False)

  return pred_int, pred_thresh, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out, roi_whitebg


def get_model_index(path, search_by):
  import re
  m = os.path.basename(path)
  found_index = re.search(search_by, m)
  f_wo_ext = os.path.splitext(path)[0]

  if found_index:
    index = found_index.start() + 1
    f_wo_ext = os.path.basename(f_wo_ext)[index:]
  else:
    f_wo_ext = os.path.basename(f_wo_ext)

  return f_wo_ext


def demo_final_seg_result(fn, res, fr_add_cont, image_crop_roi, prefix, save_im=False, save_stack_imgs=False):  # 2

  if not CLOSE_ALL:
    r_gray1, c_gray1 = res.shape
    image_crop_roi = cv2.resize(image_crop_roi, (IMAGE_SIZE, IMAGE_SIZE))
    r_rgb, c_rgb, ch = image_crop_roi.shape
    r_rgb2, c_rgb2, ch = fr_add_cont.shape

    r_comb = max(r_rgb, r_gray1, r_rgb2)
    c_comb = c_rgb + c_gray1 + c_rgb2

    comb_im = np.zeros(shape=(r_comb, c_comb, ch), dtype=np.uint8)
    comb_im[:r_rgb, :c_gray1] = res[:, :, None]  # direct output
    comb_im[:r_rgb, c_gray1:c_rgb + c_gray1] = fr_add_cont  # reverse mask
    comb_im[:r_rgb, c_rgb + c_gray1:] = image_crop_roi  # bounding box

    cv2.imshow('input-output-roi_overlay', comb_im)
    cv2.waitKey(10)

    if save_stack_imgs:
      cv2.imwrite(prefix + 'comb_' + fn + '.png', comb_im)

    if save_im:
      cv2.imwrite(prefix + '_rev_mask_' + fn + '.png', image_crop_roi)

      print 'saving image', prefix + '_rev_mask_' + fn + '.png'


def demo_stacked_n_col_images(prefix, fn, list_of_imgs, winname, save_im=False):  # 2

  width = len(list_of_imgs)
  max_r, c_comb, dim_im = 0, 0, 2

  for im in list_of_imgs:
    if len(im.shape) == 3:
      r_im, c_im, dim_im = im.shape

    else:
      r_im, c_im = im.shape
    c_comb += c_im

    if r_im > max_r:
      max_r = r_im

  r_comb = max_r

  frame_border = 1

  c_comb = c_comb + (width - 1) * frame_border

  comb_im = np.zeros(shape=(r_comb, c_comb, dim_im), dtype=np.uint8)

  white = np.zeros(shape=(r_comb, frame_border, dim_im), dtype=np.uint8)
  white2 = np.zeros(shape=(r_comb, frame_border), dtype=np.uint8)

  white.fill(255)
  white2.fill(255)

  current_column = 0

  for im in list_of_imgs:
    if len(im.shape) == 3:
      comb_im[:(im.shape[0]), current_column:current_column + im.shape[1]] = im
      if current_column + im.shape[1] < c_comb:
        comb_im[:(im.shape[0]),
        current_column + im.shape[1]:current_column + im.shape[1] + frame_border] = white
    else:
      comb_im[:(im.shape[0]), current_column:current_column + im.shape[1]] = im[:, :, None]
      if current_column + im.shape[1] < c_comb:
        comb_im[:(im.shape[0]),
        current_column + im.shape[1]:current_column + im.shape[1] + frame_border] = white2[:, :, None]

    current_column = current_column + im.shape[1] + frame_border

  if not CLOSE_ALL:
    cv2.imshow(winname, comb_im)
    cv2.waitKey(5)

  if save_im:
    cv2.imwrite(prefix + 'comb_' + fn + '.png', comb_im)

  print 'comb im shape:', comb_im.shape

  return comb_im


def save_images(prefix, fn, im):
  cv2.imwrite(prefix + 'comb_' + fn + '.png', im)
  print 'comb im shape:', im.shape


def demo_stacked_n_row_images(prefix, fn, list_of_imgs_res, winname, save_im=False):
  r_comb, c_comb, dim_im = 0, 0, 3

  for im in list_of_imgs_res:
    r_im, c_im, dim_im = im.shape
    r_comb += r_im

  frame_border = 2
  c_comb = c_im
  r_comb = r_comb + frame_border * (len(list_of_imgs_res) - 1)
  comb_im = np.zeros(shape=(r_comb, c_comb, dim_im), dtype=np.uint8)
  white = np.zeros(shape=(frame_border, c_comb, dim_im), dtype=np.uint8)
  white.fill(255)
  current_row = 0

  for im in list_of_imgs_res:
    comb_im[current_row:current_row + im.shape[0], :im.shape[1]] = im

    if current_row + im.shape[1] < r_comb:
      comb_im[current_row + im.shape[0]:current_row + im.shape[0] + frame_border, :im.shape[1]] = white

    current_row = current_row + im.shape[0] + frame_border

  cv2.imshow(winname, comb_im)

  if save_im:
    cv2.imwrite(prefix + fn + '.png', comb_im)


def EVA_IMAGE_classify_seats(read_path_im,
                             frame, frame_crop, pred_roi,
                             classified_as,
                             count_test, confMat1_TEST, confMat2_TEST, border, file_i, n_classes, fname=''):
  target = tools.get_ground_truth_label_im(read_path_im, default=False)

  print 'target label:', target

  # IMAGE
  h, w = pred_roi.shape  # pred_roi [0,255] #20181
  # print 'pred_roi shape:',h,w #net input size
  pred_roi_bin = np.float32(pred_roi > 0)
  # pred_roi_bin = pred_roi_bin / 255 #reduce to 1

  num, v = tools.count_diff_pixel_values(pred_roi_bin, h, w)
  print 'pred_roi bin num', num, v
  # print 'white pixels:', np.sum(pred_roi_bin), 'all pixels:',(h*w)

  bili = np.sum(pred_roi_bin) / (h * w)  # count of white pixel / total number of pixels in an image
  print 'proportion:', bili

  RES = classified_as
  if classified_as == 1 and bili < 0.07:
    RES = 0

  confMat1_TEST, confMat2_TEST = tools.process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop,
                                                   SAVE_CorrectClassified, SAVE_Misclassified, file_i,
                                                   target=target)

  overall_acc = do_statistics(confMat1_TEST, confMat2_TEST, count_test)
  cv2.waitKey(10)  # required for roi_seg

  return overall_acc, confMat1_TEST


create_stacked_imgs_for_paper = True


def EVA_IMAGE_seg_and_classify(MODEL_ID, bg_model, h, w,
                               in_ch=1, best_avg=0, save_res_path='', save_imgs=False, file_i=1, with_gt=False,
                               step_show=False, stop=False):
  print 'Model ID:', MODEL_ID

  res_pass_list, res_fail_list, list_of_imgs_res = [], [], []
  dices, dice_non_empty, dice_empty, dices_cad, count_test, n_classes = 0, 0, 0, 0, 0, 2
  max_dice, min_dice, avg_dice_non_empty, avg_dice_empty, overall_acc = 0, 1, 0, 0, 0
  confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
  confMat2_TEST = np.zeros((2, 2), dtype=np.float)

  read_paths_im, read_paths_m, files_im, files_m = get_read_path_and_files_for_im_m_2cl(read_from_file=False)
  # load seg model

  set_image_dim_ordering(dim_ordering='th')  #
  model = load_model(bg_model)

  if INFO_0:
    print 'loading model', bg_model

  # model.summary()
  if result_for_table:
    print 'True/False', 'No.', 'Name', 'TargetLabel', 'PredictLabel', 'Precision', \
 \
      'Top2', 'Top2_pres', 'Top3', 'Top3_pres', 'Top4', 'Top4_pres', 'Top5', 'Top5_pres', 'last', 'last_pres'


  cust_list = ['']  # can define any filename patterns
  for im_ori, ref_mask, i, read_path_im, read_path_m in zip(files_im, files_m, xrange(len(files_im)), read_paths_im,
                                                            read_paths_m):
  
    fn = os.path.basename(im_ori)[:-4]
  
    if i > -1:  # > 46 and i < 54:
      # if fn in cust_list: # 0 and i < 4:#> 46 and i < 54:
      if INFO_0:
        print 'read path:', read_path_im
  
      im_ori_view = cv2.imread(read_path_im + im_ori)
      im_crop = im_ori_view.copy()
      im_ori_view = cv2.resize(im_ori_view, (h, w))
  
      # IMAGE
      pred_int, pred_thresh,fr_add_cont, old_mask,old_roi_res, \
      new_mask, image_crop_roi, r1, r2, c1, c2, screen_out, roi_whitebg \
        = do_segment_im(model, im_ori_view, im_crop, i, h, w, in_ch, show=False, save=False)
  
    pred_thresh = new_mask.copy()  # use optimized mask for dice calc
  
    if len(pred_thresh.shape) == 3:
      pred_thresh = cv2.cvtColor(pred_thresh, cv2.COLOR_RGB2GRAY)
  
    if with_gt:
      ref_mask = cv2.resize(cv2.imread(read_path_m + ref_mask, 0), (h, w))
      ref_mask_thresh = ref_mask  # * 255.0
      thresh_ref = 0
      ref_key = 255
  
      idx = ref_mask_thresh[:, :] > thresh_ref
      ref_mask_thresh[idx] = ref_key
      pred_key = 255
  
      if search_str not in fn:
        count_test += 1
        dice = tools.calc_dice_simi(pred_thresh, ref_mask_thresh, fn, k=pred_key)
        print '\nDice for:', fn, ':', dice
        dices += dice
  
        if dice < min_dice:
          min_dice = dice
  
        if dice > max_dice:
          max_dice = dice
  
        if do_classification:
          target_label = tools.get_ground_truth_label_im(read_path_im)
          total_imgs_of_class = len(os.listdir(read_path_im))
  
          if target_label == 1:
            dice_non_empty += dice
  
            avg_dice_non_empty = dice_non_empty / total_imgs_of_class
  
          if target_label == 0:
            dice_empty += dice
            avg_dice_empty = dice_empty / total_imgs_of_class
  
    prefix = get_model_index(bg_model, '\d')
    pre_tensor_view = np.zeros((h, w, 3), np.uint8)
  
    if screen_out:
      classified_as = 1
  
      if DEBUG:
        print 'prefix:', prefix
  
      found_dig = re.search('\d', os.path.basename(bg_model))
      if found_dig:
        dig = found_dig.start()
        prefix = save_res_path + os.path.basename(bg_model)[dig:-5]
      else:
        print 'no file found'
      reduce_border = False
      if reduce_border:
        # image_crop_roi = cv2.resize(im_crop,(h,w))[r1+border:r2-border, c1-border:c2+border] #crop_y1:crop_y2, crop_x1:crop_x2
        image_crop_roi = cv2.resize(im_crop, (h, w))[r1 + border:r2 - border,
                         c1 + border:c2 - border]  # crop_y1:crop_y2, crop_x1:crop_x2
  
      if INFO_0:
        print 'image_crop_roi shape (tf ordering hwc):', image_crop_roi.shape  # tf ordering channel last
  
      if min(image_crop_roi.shape) < 1:
        # demo_final_seg_result(fn,pred_thresh,fr_add_cont,image_crop_roi,prefix,screen_out,save_im=False,save_stack_imgs=False)
  
        print 'tensor size too small'
  
        cv2.putText(fr_add_cont, 'too small ROI', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0), thickness=2)
        pre_tensor_view = np.zeros((h, w, 3), np.uint8)
  
    else:
      print 'no FG'
      classified_as = 0
      cv2.putText(fr_add_cont, 'no FG', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                  color=(0, 255, 0), thickness=2)
      pre_tensor_view = np.zeros((h, w, 3), np.uint8)
  
    if do_classification:
      overall_acc, confMat1_TEST \
        = EVA_IMAGE_classify_seats(read_path_im, im_ori_view, im_crop, pred_thresh,
                                   classified_as, count_test, confMat1_TEST, confMat2_TEST, border,
                                   i, n_classes, fn)
  
    if create_stacked_imgs_for_paper:
      s_size = (320, 320)
      if read_path_im == read_path_m:
        list_of_imgs = [cv2.resize(im_ori_view, s_size), cv2.resize(pred_int, s_size),
                        cv2.resize(new_mask, s_size),
                        ]
    
        winname = 'in-pred-oldmask-newmask'
      else:
        # list_of_imgs = [cv2.resize(im_ori_view, (320 / 2, 320 / 2)), cv2.resize(pred_int, s_size),
        #               cv2.resize(old_mask, (320 / 2, 320 / 2)), cv2.resize(fr_add_cont, s_size)]
        list_of_imgs = [cv2.resize(im_ori_view, s_size), cv2.resize(ref_mask, s_size), cv2.resize(pred_int, s_size),
                        cv2.resize(new_mask, s_size),
                        cv2.resize(fr_add_cont, s_size),
                        cv2.resize(roi_whitebg, s_size), cv2.resize(pre_tensor_view, s_size)]
    
        winname = 'in-gt-pred-newmask-cont-pretensor'
    
      print 'save path:', prefix, fn
    
      stacked_imgs = demo_stacked_n_col_images(prefix, fn, list_of_imgs, winname, save_im=True)
      list_of_imgs_res.append(stacked_imgs)
    
    # cv2.imwrite(save_res_path + 'stack_'+ os.path.basename(files_im[i]), stacked_imgs)
    
    
    if step_show and not CLOSE_ALL and not stop:
      k = cv2.waitKey(30) & 0xFF
      while True and not stop:
        if k == ord('n'):
          print 'add to fail_list:', fn
    
          res_fail_list.append(read_path_im + os.path.basename(files_im[i]))
          # res_fail_list.append(files[i])
          break
    
        elif k == ord('y'):
          print 'add to pass_list:', fn
          res_pass_list.append(read_path_im + os.path.basename(files_im[i]))
          save_imgs = False
    
          if save_imgs:
            cv2.imwrite(save_res_path + 'stack_' + os.path.basename(files_im[i]), stacked_imgs)
          break
    
        elif k == ord('q'):  # ESC
          stop = True
          break
    
        else:
          k = cv2.waitKey(30) & 0xFF
          if k != 255:
            print 'k:', k  # 81-l, 83-r, 82-u, 84-d
      if cv2.waitKey(1) & 0xFF == ord('q'):
        print 'key interrupt'
        break
    
  print_res_list = False
  
  if print_res_list:  #
    print bg_model, 'test package:', read_path_im
    print 'res_fail_list=', res_fail_list, '\nres_pass_list=', res_pass_list
    print 'num of fail:', len(res_fail_list), '\nnum of pass:', len(res_pass_list)
  
  def save_to_file(file, lines):
    lines = '\n'.join(lines)
    with open(file, 'w') as f:
      f.writelines(lines)
  
  
  save_eva_to_file = False
  
  if save_eva_to_file:
    keyword = os.path.basename(os.path.normpath(read_path_im))
  
    save_to_file('../Test_Images/img_list/fail_' + keyword + '_' + os.path.basename(bg_model)[:-5] + '.txt',
                 res_fail_list)
  
    save_to_file('../Test_Images/img_list/pass_' + keyword + '_' + os.path.basename(bg_model)[:-5] + '.txt',
                 res_pass_list)
  
  cv2.destroyAllWindows()
  
  if count_test > 0:
    avg_dice = float(dices / count_test)
  else:
    avg_dice = 0
  
  if avg_dice > best_avg:
    best_avg = avg_dice
  
  print '\nseg avg dice:', avg_dice, 'max dice:', max_dice, ', min dice:', min_dice, ', ', bg_model
  
  if do_classification:
    print '2classifier overall_acc', overall_acc
    print tools.print_label_title()
    print '#######'
    print confMat1_TEST
    print 'seg avg_non_empty:', avg_dice_non_empty, 'seg avg_empty:', avg_dice_empty
  
  else:
    overall_acc = 0
  num_total_ims = len(list_of_imgs_res)
  print 'list imgs res:', num_total_ims
  
  # create_stacked_imgs_for_paper = True
  if create_stacked_imgs_for_paper and num_total_ims > 0 and num_total_ims < 7:
    MODEL_ID = ''
    demo_stacked_n_row_images(prefix, MODEL_ID, list_of_imgs_res, winname, save_im=True)
  
  return best_avg, bg_model, overall_acc, stop


##########################################################################################################


def main(_):
  global PROJ_DIR, IMAGE_SIZE, seg_model_search_p, save_res_path, with_gt, CLOSE_ALL
  global INFO_0, DEBUG, do_classification, result_for_table, use_pretrained_cl_model
  global LABELS, LABELS_en, maxNumSaveFiles, CorrectClassified, Misclassified
  global generate_train_data, save_ori_frame, thresh_res, search_str, MODEL_ID
  global in_ch, border, step_show

  PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
  IMAGE_SIZE = 160
  seg_model_search_p = os.path.join(PROJ_DIR, 'testbench/1/')  # path to log file
  save_res_path = os.path.join(PROJ_DIR, 'testbench/k_imgs/')  # path to save res images

  with_gt = True  # test with ground truth: True/ False
  CLOSE_ALL = False  # print Info level 0 on: True/ False
  INFO_0 = False  # print Info level 0 on: True/ False
  DEBUG = False
  ############################################################

  # settings.set_global()

  do_classification = True
  result_for_table = False
  use_pretrained_cl_model = False
  LABELS = ['t', 'f']
  LABELS_en = ['true', 'false']
  maxNumSaveFiles = 10
  CorrectClassified = ''
  Misclassified = ''
  MODEL_ID = 'M_1'
  # extra
  # Seg_MODEL_to_load = '071_02-0.03' + '.hdf5'  #

  Seg_MODEL_to_load = '718_299-0.02' + '.hdf5'  #
  # EVA_IMAGE_seg_and_class = 1
  generate_train_data = False
  save_ori_frame = True

  thresh_res = 50  #

  search_str = 'cmp'

  in_ch = 1
  border = 0
  step_show = True

  bg_models = sorted([s for s in os.listdir(seg_model_search_p)
                      if Seg_MODEL_to_load in s])

  bg_model = bg_models[-1] if len(bg_models) > 1 else bg_models[0]

  print 'bg model:', bg_model

  bg_model = seg_model_search_p + bg_model
  best_avg = 0
  seg_avg, model_name, cl_overall_acc, stop = EVA_IMAGE_seg_and_classify \
 \
    (MODEL_ID, bg_model, IMAGE_SIZE, IMAGE_SIZE, in_ch, best_avg, save_res_path=save_res_path,
     save_imgs=True,
     file_i=1, with_gt=with_gt, step_show=step_show, stop=False)

cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.app.run()

