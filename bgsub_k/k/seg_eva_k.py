import tensorflow as tf
import cv2
import numpy as np
import sys
from keras.backend import set_image_dim_ordering
from keras.models import load_model
import keras
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageOps
from functools import wraps
from random import randint
import os
import datetime
import settings  # hy: collection of global variables
import tools
import time
from sklearn import datasets
import math
import imutils
from PIL import Image  # hy: create video with images
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import re
# from background_learning_s import dice_coef_loss
# tf.python.control_flow_ops = tf #hy:for remote
# KERAS_BACKEND=tensorflow python -c "from keras import backend"
# Using TensorFlow backend.

#####################################################################################################
# activate global var
PROJ_DIR = '/home/hy/unet/'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("IMAGE_SIZE", "160", "input image size")
tf.flags.DEFINE_string("logs_dir", "../logs/", "path to logs directory")
tf.flags.DEFINE_bool('log_on', "False", "Log mode: True/ False")
tf.flags.DEFINE_string('seg_model_search_p', PROJ_DIR + 'testbench/', "path to log file")
tf.flags.DEFINE_string('save_res_path', PROJ_DIR + 'testbench/k_imgs/', "path to save res images")
tf.flags.DEFINE_bool('with_gt', "True", "test with ground truth: True/ False") #
tf.flags.DEFINE_bool('CLOSE_ALL', "True", "close all windows: True/ False")
tf.flags.DEFINE_bool('INFO_0', "False", "print Info level 0 on: True/ False")
tf.flags.DEFINE_bool('DEBUG', "False", "print Debug log: True/ False")


do_multiple_test = False  # this can only be started with prepared shell script eva.sh
############################################################
settings.set_global()
start_time = time.time()
do_6face_classification = False
result_for_table = False

#check
#add_statistic_page
#close_all
#out comment all segmodel
#save pretensor

if do_multiple_test:
	# use setup in eva.sh
	Seg_MODEL_to_load = sys.argv[1]  # + ".meta"
	print 'model load from shell:', Seg_MODEL_to_load
	if sys.argv[2] == '1':
		use_cut_bounding_box = True
	else:
		use_cut_bounding_box = False

	if sys.argv[3] == '1':
		use_seg_limit = True
	else:
		use_seg_limit = False
	MODEL_ID = sys.argv[4]
	save_res_path = sys.argv[5]
	SAVE_MUL_RES_PATH = sys.argv[5]
	print 'model id:', MODEL_ID

else:
	use_cut = False #cut or seg cut
	use_limit = True
	#MODEL_ID = 'M_cam'
	MODEL_ID = 'M_28'

	Seg_MODEL_to_load = 'No303' + '.hdf5'#  28 best seg
	


EVA_IMAGE_seg_and_class        = 1

EVALUATE_WEBCAM = 0
EVALUATE_VIDEO  = 0

generate_train_data = False
Img_255 = False #False- mul, True- bin
save_ori_frame = True
thresh_res = 50 #
search_str = 'cad'
in_ch = 1
border = 0
step_show = False
#manual
#bg_model = '../testbench/seg_mul/all_real/rgb00.hdf5'
bg_model2 = '../testbench/bg_door_tree/weights_door_tree_pathches_396.hdf5' #weights_tree_pathches_266 weights_no_full_433_696_p05_g
classifier_model = '../testbench/6classes/' + 'model_3conv_GD360_h184_w184_c6_b20_1_0.98_II_0.75_III_0.63-12811' + '.meta'  #top6cl:0.54,clear
n_hidden = 360  # n_hidden = classifier_model.split('conv')[1][0:3]
#n_hidden = 128*4  # n_hidden = classifier_model.split('conv')[1][0:3]
do_active_fields_test = 0
#dropout = [0.3, 0.3, 0.5, 0.5]  # 3,4,5,5
dropout = [0.15, 0.25, 0.4, 0.5, 1, 0.4, 0.25, 0.15, 0.15]  # 1st
dropout_1s = [1] * len(dropout)

##########

TEST_CONV_OUTPUT = False

SAVE_Misclassified = 0
SAVE_CorrectClassified = 0

###########

LABEL_LIST = '../FileList.txt'
LABEL_PATH = settings.data_label_path

LABEL_LIST_TEST = '../FileList_TEST1.txt'
LABEL_PATH_TEST = settings.test_label_path8


######################################################################################################
def get_read_path_and_files_for_im(read_from_file=False):
	if read_from_file:
		files = []

		file_with_gt = False
		if file_with_gt:
			# read file containing filename and ground truth data
			with open('../Test_Images/full_frames_sun/seg_ground_truth/pass_single_gt.txt', 'r') as f:
				lines = [r.split()[0] for r in f]
		else:
			lines = open(
				'../Test_Images/img_list/fail_full_frames_clear_w_table_patches_136top.txt').read().splitlines()

		read_path = os.path.dirname(lines[0]) + '/'
		if FLAGS.INFO_0:
			print 'read_path', read_path
		for line in lines:
			files.append(os.path.basename(line))

	else:
		read_path = '../'
		read_path = '../Test_Images/MA/from_videos/machineroom_40_44/'  ##machineroom_40_44 #outside_39_46_47  #office_41_43
		#read_path = '../tmp/bg//bg_MA_exp/5/'  #machineroom_40_44 #outside_39_46_47  #office_41_43
		read_path = '../Test_Images/MA/test_represent/images/'  #machineroom_40_44 #outside_39_46_47  #office_41_43
		read_path = '../Test_Images/MA/test_represent/blue_red_pair/'
		files = [s for s in os.listdir(read_path)]
		#files = files[0:40]
		read_paths_im = []
		for i in xrange(len(files)):
			read_paths_im.append(read_path)


	if FLAGS.INFO_0:
		print 'num of files:', len(files), ',  files[0]:', files[0]
	return read_paths_im, files

def get_read_path_and_files_for_im_m(read_from_file=False,data_path=''):#represent
	if read_from_file:
		files = []

		file_with_gt = False
		if file_with_gt:
			# read file containing filename and ground truth data
			with open('../Test_Images/full_frames_sun/seg_ground_truth/pass_single_gt.txt', 'r') as f:
				lines = [r.split()[0] for r in f]
		else:
			lines = open(
				'../Test_Images/img_list/fail_full_frames_clear_w_table_patches_136top.txt').read().splitlines()

		read_path = os.path.dirname(lines[0]) + '/'
		if FLAGS.INFO_0:
			print 'read_path', read_path
		for line in lines:
			files.append(os.path.basename(line))

	else:
                read_path_im = os.path.join(data_path, 'images/')
                read_path_m = os.path.join(data_path, 'labels/')
		
		files_im = sorted([s for s in os.listdir(read_path_im) ])
		files_m = sorted([s for s in os.listdir(read_path_m) ])
		
		files_im = files_im[0:10]
		files_m = files_m[0:10]

		read_paths_im, read_paths_m = [], []
		for i in xrange(len(files_im)):
			read_paths_im.append(read_path_im)
			read_paths_m.append(read_path_m)

	# for im in res_fail_list:
	#	files.append(read_path + im)
	if FLAGS.INFO_0:
		print 'num of files:', len(files_im), ',  files[0]:', files_im[0]
	return read_paths_im,read_paths_m, files_im, files_m

#Load 6cl
def get_read_path_and_files_for_im_m_6cl(read_from_file=False):  # represent
	if read_from_file:
		files = []
		
		file_with_gt = False
		if file_with_gt:
			# read file containing filename and ground truth data
			with open('../Test_Images/full_frames_sun/seg_ground_truth/pass_single_gt.txt', 'r') as f:
				lines = [r.split()[0] for r in f]
		else:
			lines = open(
				'../Test_Images/img_list/fail_full_frames_clear_w_table_patches_136top.txt').read().splitlines()
		
		read_path = os.path.dirname(lines[0]) + '/'
		if FLAGS.INFO_0:
			print 'read_path', read_path
		for line in lines:
			files.append(os.path.basename(line))
	
	else:
		#folders = settings.LABELS
		#folders = ['/']
		total_files_im,total_files_m = [], []
		all_read_path_im, all_read_path_m = [], []
		folders = ['vorn/','hinten/','links/','rechts/','unten/','oben/'] #Paper order
		#folders = ['hinten/'] #Paper order
		#folders = ['2_bad_recognized/']
		for folder in folders:
			read_path_im = PROJ_DIR + '/Test_Images/MA/test_represent/6classifier_seg/im/' + folder
			read_path_m = PROJ_DIR + '/Test_Images/MA/test_represent/6classifier_seg/m/' + folder


					
			files_im = sorted([s for s in os.listdir(read_path_im)])# if 'offi' in s])
			files_m = sorted([s for s in os.listdir(read_path_m)])# if 'offi' in s])
			
			#files_im = files_im[0:20]
			#files_m = files_m[0:20]
			
			#clear
			#files_im = sorted([s for s in os.listdir(read_path_im) if 'frame_crop1620' in s or 'links_clear2_3' in s or 'links_clear_140' in s
			#                or 'oben_clear_15' in s or 'oben_ts_14700' in s or 'rechts_clear_1140' in s or 'rechts_clear2_26' in s
			#                 or 'unten_clear_10' in s or 'unten_clear_28' in s or 'vorn_ts_540' in s or 'vorn_ts_792' in s])  # if 'offi' in s])
			#files_m = sorted([s for s in os.listdir(read_path_m) if 'frame_crop1620' in s or 'links_clear2_3' in s or 'links_clear_140' in s
			#                 or 'oben_clear_15' in s or 'oben_ts_14700' in s or 'rechts_clear_1140' in s or 'rechts_clear2_26' in s
			#                   or 'unten_clear_10' in s or 'unten_clear_28' in s or 'vorn_ts_540' in s or 'vorn_ts_792' in s])  # if 'offi' in s])
			
			
			
			#files_im = sorted([s for s in os.listdir(read_path_im) if 'offi' in s]) #set scene
			#files_m = sorted([s for s in os.listdir(read_path_m) if 'offi' in s])
			
			total_files_im = total_files_im + files_im
			total_files_m = total_files_m + files_m
			
			for i in xrange(len(files_im)):
				all_read_path_im.append(read_path_im)
				all_read_path_m.append(read_path_m)
			
	
	if FLAGS.INFO_0:
		print 'num of files:', len(files_im), ',  files[0]:', files_im[0]
	return all_read_path_im, all_read_path_m, total_files_im, total_files_m

def add_colorOverlay(img_grayscale, mask):
  colorOverlay = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB)
  colorOverlay[:, :, 2] = mask
  return colorOverlay

def demo_result_imgs(bg_model,file1, file2, file3, frame_i=1, save_file=False, demo=False): #1
	if save_file:
		cv2.imwrite('../testbench/frame_pred_' + bg_model[:-5] + '%03d.jpg' % frame_i, np.uint8(file1))
		cv2.imwrite("../testbench/frame_combined_%03d.jpg" % frame_i, file2)
		cv2.imwrite("../testbench/frame_color_%03d.jpg" % frame_i, file3)
	if demo:
		title = 'direct_output'
		cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
		cv2.putText(np.uint8(file1), 'No. ' + str(frame_i+1), org=(320 / 10, 320 / 8),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
		            color=(0, 255, 255), thickness=2)
		if not FLAGS.CLOSE_ALL:
			cv2.imshow(title, np.uint8(file1))
			cv2.imshow('input_direct_output', file2)
		#print 'seg_feature uint8\n',np.uint8(file1)
		res_pass_list, res_fail_list = [], []
		k = cv2.waitKey(30) & 0xFF
		while True:
			if k == ord('n'):
				print 'add to fail_list'
				res_fail_list.append(frame_i)
				# res_fail_list.append(files[i])
				break
			elif k == ord('y'):
				print 'add to pass_list'
				res_pass_list.append(frame_i)
				save_imgs = True
				if save_imgs:
					cv2.imwrite('../classified/MA_k/pass_cv_' + str(frame_i) + '.png', file2)
					#misc.imsave('../classified/MA_1/pass_misc_' + str(im_i) + '.png', im_ori_cv2)
				break
			elif k == ord('q'):  # ESC
				break
			else:
				k = cv2.waitKey(30) & 0xFF
				if k != 255:
					print 'k:', k  # 81-l, 83-r, 82-u, 84-d

#https://en.wikipedia.org dice

def get_bounding_box(conture, img=None,draw_box=False):
	""" Calculates the bounding box of a ndarray"""
	# get approx, return index
	# epsilon = 0.1 * cv2.arcLength(x, True)
	# approx_box = cv2.approxPolyDP(x, epsilon, True)
	# print 'app box', approx_box  # Min [[[ 56  85]]  [[318 231]]]
	# leftpointX = approx_box[0][0][0]
	# print 'app box 2', leftpointX  # Min [[[ 56  85]] Max [[318 231]]]
	# approx_box_s = int(0.9*approx_box)
	# print 'app box s',approx_box_s
	
	# get rectangle
	x, y, w, h = cv2.boundingRect(conture)  # x,y: top-left coordinate
	# draw rectangle
	if draw_box:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.waitKey(2)
	return (x, y, w, h)

def get_bounding_box_with_limit(bin_im_cv,draw_box=False):
	im_pil = bin_im_cv.copy()
	if len(bin_im_cv.shape) == 3:
		im_pil = cv2.cvtColor(bin_im_cv, cv2.COLOR_BGR2RGB)
	im = Image.fromarray(im_pil)
	im = im.convert('LA')  # convert Image object image to gray
	pix = im.load()
	w, h = im.size
	if FLAGS.INFO_0:
		print 'w,h:', w, h
	wh_count, t, b, l, r = 0, 0, 0, 0, 0
	ts, bs, ls, rs = [], [], [], [] #tmp
	t_counts, b_counts, l_counts, r_counts = [], [], [], []
	limit = 1
	set_break = False  # t
	for y in xrange(h):
		wh_count = 0
		for x in xrange(w):
			if pix[x, y] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				t_counts.append(wh_count)
	
	limit = np.median(sorted(t_counts))
	if FLAGS.INFO_0:
		print 't_median t', limit
	for y in xrange(h):
		wh_count = 0
		if set_break:
			break
		for x in xrange(w):
			# print pix[x,y]
			if pix[x, y] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				t = y
				set_break = True
				break
	
	limit = 1
	for y in xrange(h):
		wh_count = 0
		for x in xrange(w):
			# print pix[w-x-1,h-y-1]
			if pix[w - x - 1, h - y - 1] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				b_counts.append(wh_count)
	
	limit = np.median(sorted(b_counts))
	if FLAGS.DEBUG:
		print 'limit b', limit
	# b_mean = np
	set_break = False  # b
	for y in xrange(h):
		wh_count = 0
		if set_break:
			break
		for x in xrange(w):
			# print pix[w-x-1,h-y-1]
			if pix[w - x - 1, h - y - 1] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				b = h - y - 1
				set_break = True
				break
	
	###################################
	limit = 1
	for x in xrange(w):
		wh_count = 0
		for y in xrange(h):
			# print pix[w-x-1,h-y-1]
			if pix[x, y] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				l_counts.append(wh_count)
	limit = np.median(sorted(l_counts))
	
	set_break = False  # l
	for x in xrange(w):
		wh_count = 0
		if set_break:
			if FLAGS.DEBUG:
				print 'x:', wh_count
			break
		for y in xrange(h):
			# print pix[w-x-1,h-y-1]
			if pix[x, y] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				l = x
				if FLAGS.DEBUG:
					print wh_count
				set_break = True
				break
	
	################################
	limit = 1
	for x in xrange(w):
		wh_count = 0
		for y in xrange(h):
			# print pix[w-x-1,h-y-1]
			if pix[w - x - 1, h - y - 1] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				r_counts.append(wh_count)
	
	limit = np.median(sorted(r_counts))
	
	set_break = False  # r
	for x in xrange(w):
		wh_count = 0
		if set_break:
			break
		for y in xrange(h):
			# print pix[w-x-1,h-y-1]
			if pix[w - x - 1, h - y - 1] != (255, 255):
				wh_count += 1
			if wh_count > limit:
				r = w - x - 1
				set_break = True
				break
	
	if FLAGS.DEBUG:
		print 't,b,l,r', t, ',', b, ',', l, ',', r
	x, y, bw, bh = l, t, (r - l), (b - t)
	if draw_box:
		cv2.rectangle(bin_im_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return x,y,bw,bh

def get_rgb_roi(roi_bin, base_img, w, h):
	roi_gray = cv2.cvtColor(roi_bin, cv2.COLOR_BGR2RGB)
	roi_PIL = Image.fromarray(roi_gray)
	datas_obj = roi_PIL.getdata()
	
	fr = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
	fr_PIL = Image.fromarray(fr)
	datas_fr = fr_PIL.getdata()
	
	new_im = Image.new("RGB", (w, h))  ## default black!
	new_im.paste((255, 255, 255), (0, 0, w, h))  ##
	datas_roi = new_im.convert("RGBA")
	
	newData = []
	
	def white(data_obj):
		if data_obj[0] == 255 and data_obj[1] == 255 and data_obj[2] == 255:
			white = True
		else:
			white = False
		return white
	
	for data_obj, data_fr in zip(datas_obj, datas_fr):
		if white(data_obj):
			newData.append((data_fr[0], data_fr[1], data_fr[2]))
		else:
			newData.append((255, 255, 255))
	
	datas_roi.putdata(newData)
	pil_image = datas_roi.convert('RGB')
	open_cv_image = np.array(pil_image)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	return open_cv_image

def get_roi_with_white_bg(blackbg, base_img, cont, w, h, set_limit=False):
	"""
  Get ROI with white background
  :param roi_bin: image to be converted to binary -
  :param base_img: RGB input
  :param cont: contours
  :param set_limit: use limit for cutting boundary
  Available options:
  set_limit = True/ False
  """
	min_size = 4
	open_cv_image_wh = get_rgb_roi(blackbg, base_img, w, h)
	ret, open_cv_image = cv2.threshold(blackbg, 254, 255, cv2.THRESH_BINARY_INV)
	if not set_limit:
		
		x, y, bw, bh = get_bounding_box(cont, open_cv_image, draw_box=False)
		
		if bw > min_size and bh > min_size:
			screen_out = True
	else:
		screen_out = False
		x, y, bw, bh = get_bounding_box_with_limit(open_cv_image, draw_box=False)
		if bw > min_size and bh > min_size:
			screen_out = True
	
	return x, y, bw, bh, screen_out, open_cv_image_wh

def get_roi_with_white_bg_cut(roi_bin, base_img, w, h, cont,factor=1,set_limit=True):
	#cv2.imwrite('../roi.png',roi)
	ret,mask = cv2.threshold(roi_bin,254,255,cv2.THRESH_BINARY_INV)
	if set_limit:
		x, y, box_w, box_h = get_bounding_box_with_limit(mask,draw_box=False)  # x,y:top-left coord mask
	else:
		x, y, box_w, box_h = get_bounding_box(cont,draw_box=False)  # x,y:top-left coord mask

	if FLAGS.DEBUG:
		print 'get_white_cut,size',roi_bin.shape,'base size',base_img.shape
	if box_w > 3 and box_h > 3:
		screen_out = True
		if FLAGS.INFO_0:
			print 'x, y, w, h', x, y, box_w, box_h
	else:
		screen_out = False
		x, y, box_w, box_h = 0, 0, 0, 0
	return x, y, box_w, box_h, screen_out

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
		
		cv2.drawContours(fr_add_cont, [largest_areas[-1]], 0, (255, 255, 255, 255), -1)  # 1
		cv2.drawContours(blackbg, [largest_areas[-1]], 0, (255, 255, 255, 255), -1)  # 1
		
		if use_cut:
			
			x, y, bw, bh, screen_out = get_roi_with_white_bg_cut(blackbg, fr_ori_int8, largest_areas[-1],
			                                                     set_limit=use_limit)
			if screen_out:
				r1, r2, c1, c2 = y, y + bh, x, x + bw
				new_roi_res = roi_res[r1:r2, c1:c2]
				new_mask[r1:r1 + bh, c1:c1 + bw, :] = blackbg[r1:r2, c1:c2]  # better
				whitebg[r1:r1 + bh, c1:c1 + bw, :] = new_roi_res
				if not FLAGS.CLOSE_ALL and FLAGS.DEBUG:
					cv2.imshow('new_roi', new_roi_res)
					cv2.imshow('old_mask', blackbg)
					cv2.imshow('new_mask', new_mask)
			else:
				x, y, bw, bh = 0, 0, 0, 0
				r1, r2, c1, c2 = y, y + bh, x, x + bw
				new_mask = np.zeros((w, h, 3), np.uint8)
		
		else:
			x, y, bw, bh, screen_out, open_cv_image = get_roi_with_white_bg(blackbg, fr_ori_int8, largest_areas[-1], w, h,
			                                                                set_limit=use_limit)
			if screen_out:
				r1, r2, c1, c2 = y, y + bh, x, x + bw
				new_roi_res = open_cv_image[r1:r2, c1:c2]
				new_mask[r1:r1 + bh, c1:c1 + bw, :] = blackbg[r1:r2, c1:c2]  # better
				whitebg[r1:r1 + bh, c1:c1 + bw, :] = open_cv_image[r1:r2, c1:c2]
			else:
				x, y, bw, bh = 0, 0, 0, 0
				r1, r2, c1, c2 = y, y + bh, x, x + bw
				new_mask = np.zeros((w, h, 3), np.uint8)
			
			if FLAGS.DEBUG:
				cv2.imshow('old_roi', old_roi_res)
				cv2.imshow('new_roi', new_roi_res)
				cv2.imshow('old_mask', blackbg)
				cv2.imshow('new_mask', new_mask)
				cv2.waitKey()
	
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


#MA display
def get_classify_result(sess, test_image, test_labels, im_i, frame, frame_crop,
                        crop_y1, crop_y2, crop_x1, crop_x2, border,screen_out=False,fname='',target=0):
	# print 'frame shape', frame.shape[0], frame.shape[1]
	######################### Tensorflow
	if not result_for_table:
		print '\nNo.', im_i+1, ' file:', fname
	#test_image= test_image[crop_y1:crop_y2,crop_x1:crop_x2]
	
	batch_xs, batch_ys = test_image, test_labels
	
	crop_y1, crop_y2, crop_x1, crop_x2 = int(frame_crop.shape[0]*crop_y1/320), int(frame_crop.shape[0]*crop_y2/320),\
	int(frame_crop.shape[1] * crop_x1/320), int(frame_crop.shape[1]*crop_x2/320)
	
	# print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
	dropout_cl = [0,0,0,0]
	dropout_1s_cl = [1] * len(dropout_cl)
	
	output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s_cl})
	
	# print("Output for external=",output)
	output = tools.convert_to_confidence(output)  #
	np.set_printoptions(precision=3)
	if FLAGS.DEBUG:
		print 'output', output
	rank_outputs = sorted(range(len(settings.LABELS)), key=output[0].__getitem__)
	
	RES = np.argmax(output)  # hy index starting from 0, 0-5 corresponding predicted label 1-6

	label_pred_str = settings.LABELS_en[RES][:-1]
	if target == RES:
		#print 'RES:', RES, 'target', target
		tp_color = (0,0,255)
	else:
		tp_color = (0,0,0)
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

	if FLAGS.INFO_0:
		print prob_str2, prob_str3, prob_str4, prob_str5, prob_str6
	
	# hy: for sub-classes
	# label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
	# RES_sub_to_face = class_label #hy added
	# print "target, predict =", target, ', ', RES  # hy
	frame = frame_crop
	if FLAGS.DEBUG:
		print 'frame shape(classify demo):', frame.shape
	if frame.shape[0] > 330:
		frontsize, frontsize_no,frontsize_stat,thickness, thickness_no = 2, 1,   1.5, 4, 2
	else:
		frontsize, frontsize_no,frontsize_stat,thickness, thickness_no = 0.4,0.5,1.5,1, 1
		
	if do_6face_classification:##---
		print 'No.',im_i+1, fname,target, label_pred_str, prob_str, label_pred_str2, prob_str2, label_pred_str3, \
			prob_str3, label_pred_str4, prob_str4, label_pred_str5, prob_str5, label_pred_str6, prob_str6
	demo_window_width = 600  # 600 * 1080 / 1920
	add_statistic_page = True
	stat = np.zeros((600,600,3),np.uint8)
	stat.fill(255)
	txt_col1 = 1#13#int(demo_window_width*0.022)
	txt_col2 = 250#93#int(demo_window_width* 0.155)
	if FLAGS.IMAGE_SIZE == 160:
		rows = [60,120,170,220,270]
	else:
		rows = [60,120,150,180,210]#need to find other appropriate values
		
	if screen_out and not FLAGS.CLOSE_ALL:
		cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=thickness_no)
		#frame_demo = imutils.resize(frame, width=600)
		cv2.putText(frame, "predicted top 1: " + label_pred_str + ' confid.:' + prob_str,
		            org=(frame.shape[1] / 5, int(frame.shape[0] * 0.1)),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(0, 255, 0), thickness=thickness)
		if add_statistic_page:#int(demo_window_width * 0.1)
			cv2.putText(stat, "1: " + label_pred_str,
		            org=(txt_col1, rows[0]),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=tp_color, thickness=thickness)
			cv2.putText(stat, prob_str,
					org=(txt_col2, rows[0]),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)

		###########################################################################################################
		cv2.putText(frame, "predicted top 2: " + label_pred_str2 + ' confid.:' + prob_str2,
		            org=(frame.shape[1] / 5, int(frame.shape[0] * 0.20)),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
		if add_statistic_page:
			cv2.putText(stat, "2: " + label_pred_str2,
		            org=(txt_col1, rows[1]),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
			cv2.putText(stat, prob_str2,
					org=(txt_col2, rows[1]),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)

		###########################################################################################################
		cv2.putText(frame, "predicted top 3: " + label_pred_str3 + ' confid.:' + prob_str3,
		            org=(frame.shape[1] / 5, int(frame.shape[0] * 0.25)),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)

		if add_statistic_page:
			cv2.putText(stat, "3: " + label_pred_str3,
		            org=(txt_col1, rows[2]),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
			cv2.putText(stat, prob_str3,
					org=(txt_col2, rows[2]),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)

		###########################################################################################################

		cv2.putText(frame, "predicted top 4: " + label_pred_str4 + ' confid.:' + prob_str4,
		            org=(frame.shape[1] / 5, int(frame.shape[0] * 0.30)),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
		if add_statistic_page:
			cv2.putText(stat, "4: " + label_pred_str4,
		            org=(txt_col1, rows[3]),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
			cv2.putText(stat, prob_str4,
					org=(txt_col2, rows[3]),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
		

		###########################################################################################################
		cv2.putText(frame, "predicted top 5: " + label_pred_str5 + ' confid.:' + prob_str5,
		            org=(frame.shape[1] / 5, int(frame.shape[0] * 0.35)),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
		if add_statistic_page:
			cv2.putText(stat, "5: " + label_pred_str5,
		            org=(txt_col1, rows[4]),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
			cv2.putText(stat, prob_str5,
					org=(txt_col2, rows[4]),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)


		###########################################################################################################
		Print_ALL = False
		if Print_ALL:
			cv2.putText(frame, "predicted top 6: " + label_pred_str6 + ' confid.:' + prob_str6,
						org=(frame.shape[1] / 5, int(frame.shape[0] * 0.4)),
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize, color=(255, 0, 0), thickness=thickness)
			if add_statistic_page:
				cv2.putText(stat, "6: " + label_pred_str6,
						org=(txt_col1, 240),
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)
				cv2.putText(stat, prob_str6,
						org=(txt_col2, 240),
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_stat, color=(0, 0, 0), thickness=thickness)

		# print 'video.get',str(video.get(1))
		#frame.shape[1] *9/ 12 for w, frame.shape[0] *11/ 12 for h
		
		frame_demo = imutils.resize(frame, width=demo_window_width) #cannot use frame because it should be reserved for receiving next input
		demo_window_height = int(demo_window_width* frame.shape[0]/frame.shape[1])
		cv2.imshow('stat',stat)
		if FLAGS.DEBUG :
			print 'demo height', demo_window_height
		cv2.putText(frame_demo, 'No. ' + str(im_i+1), org=(10, demo_window_height -16),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=frontsize_no, color=(0, 255, 0), thickness=thickness_no)
		
		if FLAGS.INFO_0 or EVALUATE_WEBCAM==1:
			cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
			cv2.imshow("Demo", frame_demo)
			
	else:
		frame_demo = imutils.resize(frame, width=demo_window_width)
		
	return int(RES),frame_demo,stat


def get_tensor(im_i, pre_tensor, n_classes, cvtcolor, screen_out):
	# tensorImgIn = cv2.imread('../testbench/frame_color_tmp.jpg')
	# transform color and size to fit trained classifier model
	if screen_out and min(pre_tensor.shape)>0:
		pre_tensor_view = pre_tensor.copy()
		putText = False
		if putText:
			cv2.putText(pre_tensor_view, 'No. ' + str(im_i+1), org=(10, 20),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
		            color=(0, 255, 0), thickness=2)
		if not FLAGS.CLOSE_ALL and FLAGS.DEBUG:
			cv2.namedWindow('pre_tensor', cv2.WINDOW_NORMAL)
			cv2.imshow('pre_tensor', pre_tensor_view)
	
	if cvtcolor:
		pre_tensor = cv2.cvtColor(pre_tensor, cv2.COLOR_BGR2GRAY)
	
	# in case gray image as test image, no need to cvt
	
	if FLAGS.IMAGE_SIZE < 320:
		test_image = cv2.resize(pre_tensor, (settings.h_resize, settings.w_resize),interpolation=cv2.INTER_CUBIC)
		#test_image = cv2.resize(pre_tensor, (settings.h_resize, settings.w_resize))
	else:
		test_image = cv2.resize(pre_tensor, (settings.h_resize, settings.w_resize))
	test_image = np.asarray(test_image, np.float32)
	
	tensorImgIn = test_image.reshape((-1, test_image.size))
	tensorImgIn = np.expand_dims(np.array(tensorImgIn), 2).astype(np.float32)
	tensorImgIn = tensorImgIn / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
	
	test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
	return tensorImgIn, test_labels, pre_tensor_view

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

#webcam
def do_segment(model, im_crop_color, im_i, h, w, in_ch, show_bbox=False):
	res_pass_list, res_fail_list = [], []
	####### convert into the shape for seg model input
	if FLAGS.INFO_0:
		print w, h  # 320,320

	im_crop = cv2.resize(cv2.cvtColor(im_crop_color, cv2.COLOR_BGR2GRAY), (h, w))
	im_crop = np.float32(im_crop.reshape(h, w))
	im_crop = im_crop / 255.0
	im_crop = tools.reduce_mean_stdev(im_crop, print_val=False)

	image_k_tensor = np.zeros((3, 1, h, w))  # 3,1,320,320
	image_k_tensor[1, :, :, :] = im_crop
	image_k_tensor = image_k_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape
	# debug
	# print '2-shape of test images', images.shape  # (1, 1, 320, 320)

	######################
	images_original = image_k_tensor.copy()
	if FLAGS.DEBUG:
		print 'image_tensor shape:', image_k_tensor.shape
	image_k_tensor = image_k_tensor.reshape(-1,in_ch,h,w)
	#image_k_tensor = np.transpose(image_k_tensor, (0, 2, 1, 3)) #tmp to check

	pred = model.predict(image_k_tensor, batch_size=1)  # webcam
	if FLAGS.INFO_0:
		print 'Test image', im_i, ", Min,Max: %f %f" % (np.min(pred), np.max(pred))
	if (np.min(pred) == np.max(pred)):
		print 'no file got'

	pred_int = pred[0, 0, :, :].reshape((h, w))  #
	pred_255 = pred_int * 255  #
	#############################################################################
	# tmp_time0=time.time() #time for a loop 5.88762402534,5.81136989594 start from here
	# webcam
	fr_add_cont,old_mask,old_roi_res,new_mask,pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg = ROI(pred_255, im_crop_color, h, w, im_i=im_i,
																  save_file=False)

	# save_imgs = False
	if show_bbox and screen_out:
		# screen_out = False
		prefix = save_res_path + get_model_index(bg_model, search_by='-')
		# tmp
		# prefix = save_res_path +  '6cl_44_exp2_blank_test1_04-0.18_current_best'
		if FLAGS.INFO_0:
			print 'prefix:', prefix
	return pred_int, pred_255, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out, roi_whitebg


def do_segment_video(model,im_crop_color, im_i, h, w,in_ch,show_bbox=False):
	res_pass_list, res_fail_list = [], []
	####### convert into the shape for seg model input
	if FLAGS.INFO_0:
		print w, h #320,320
	im_crop = cv2.resize(cv2.cvtColor(im_crop_color,cv2.COLOR_BGR2GRAY), (h, w))
	im_crop = np.float32(im_crop.reshape(h, w))
	im_crop = im_crop / 255.0
	im_crop = tools.reduce_mean_stdev(im_crop,print_val=False)

	image_tensor = np.zeros((3, 1, h, w))  # 3,1,320,320
	image_tensor[1, :, :, :] = im_crop
	if FLAGS.INFO_0:
		print 'image_tensor.shape1:', image_tensor.shape #1
	
	image_tensor = image_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape
	# debug
	# print '2-shape of test images', images.shape  # (1, 1, 320, 320)

	######################
	images_original = image_tensor.copy()
	#image_tensor = np.transpose(image_tensor,(0,2,1,3))
	#print 'image_tensor.shape2:', image_tensor.shape #1
	
	for i in range(0, image_tensor.shape[0]):
		start = time.time()
		#ch = 3
		pred = model.predict(image_tensor[i, :, :, :].reshape(1, in_ch, h, w), batch_size=1)  #video
		#print 'model direct output,reshape:',result
		end = time.time()
		if FLAGS.INFO_0:
			print 'time elapsed for calc seg feature', (end - start), 's'  # 1.24031400681 s
			print 'Test image', im_i, ", Min,Max: %f %f" % (np.min(pred), np.max(pred))
		if (np.min(pred) == np.max(pred)):
			print 'no file got'
			break
		
		# debug
		# print 'result shape', (result.shape)
		pred_int = pred[0, 0, :, :].reshape((h, w))  #
		pred_255 = pred_int * 255  #

		
		# debug
		# cv2.imwrite('../testbench/frame_res_tmp.jpg', np.uint8(res))
		if FLAGS.INFO_0:
			print '# show segment result for frame', im_i

	#############################################################################
	# tmp_time0=time.time() #time for a loop 5.88762402534,5.81136989594 start from here
	# CLASSIFICATION
	#do_segmentVIDEO
	fr_add_cont,old_mask,old_roi_res,new_mask,pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg = ROI(pred_255, im_crop_color, h, w, im_i=im_i, save_file=False)

	# save_imgs = False
	if show_bbox and screen_out:
		# screen_out = False
		prefix = FLAGS.save_res_path + get_model_index(Seg_MODEL_to_load, search_by='-')
		#tmp
		#prefix = save_res_path +  '6cl_44_exp2_blank_test1_04-0.18_current_best'
		print 'prefix:', prefix
		fn = ''
		#VIDEO
		demo_final_seg_result(fn, pred_255, fr_add_cont, pre_tensor, prefix, screen_out, save_stack_imgs=False)

	# time for ROI 5.74937677383
	# tmp_time0 = time.time() #time for a loop 0.0359718799591 start from here
	# print 'time for write',tmp_time2-tmp_time1 # 0.00110197067261, 7.58029007912(no write), 7.33648395538(use write)
	return pred_int,pred_255,fr_add_cont,old_mask,old_roi_res,new_mask,pre_tensor, r1, r2, c1, c2,screen_out,roi_whitebg

def do_segment_im(model, im_crop_color, im_crop, im_i, h, w,in_ch,show=False,save=False):
	####### convert into the shape for seg model input

	if FLAGS.INFO_0:
		print w, h
	if in_ch == 1:
		im_crop = imutils.resize(cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY), width=w)
		im_crop = cv2.resize(im_crop, (h, w))
		im_crop = np.float32(im_crop.reshape(h, w)) / 255.0
		#im_crop = im_crop / 255.0

		image_k_tensor = np.zeros((3, 1, h, w))  # 3,1,320,320 in theano ordering
		image_k_tensor[1, :, :, :] = im_crop
		image_k_tensor = image_k_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape
		image_view = cv2.resize(im_crop_color, (h, w))
		if FLAGS.INFO_0:
			print 'image_tensor 0:', image_k_tensor.shape  #'th': (ch, h, w),  'tf': (h, w, ch)
		#image_k_tensor = np.transpose(image_k_tensor, (0, 2, 1, 3))
		#print 'image_tensor 1:', image_k_tensor.shape


	else:
		#im_crop = imutils.resize(cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY), width=w)
		im_crop = cv2.resize(im_crop_color, (h, w)) / 255.0

		im_crop = np.rollaxis(im_crop, axis=2, start=0) #change order of dimenstions
		image_k_tensor = np.zeros((3, 3, h, w))  #
		image_k_tensor[1, :, :, :] = im_crop
		image_k_tensor = image_k_tensor[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape

		image_view = cv2.resize(im_crop_color,(h,w))

	# debug
	# print '2-shape of test images', images.shape  # (1, 1, 320, 320)

	######################
		
	image_k_tensor = tools.reduce_mean_stdev(image_k_tensor, print_val=False)
	res_pass_list, res_fail_list = [], []

	#print 'images.shape[0]:', images.shape[0] #1
	for i in range(0, image_k_tensor.shape[0]):
		start = time.time()

		pred = model.predict(image_k_tensor[i, :, :, :].reshape(1, in_ch, h, w), batch_size=1)  # MA_im

		if FLAGS.DEBUG:
			print 'model direct output,reshape:',pred
		end = time.time()
		if FLAGS.INFO_0:
			print 'time elapsed for calc seg feature', (end - start), 's'  # 1.24031400681 s
			print 'Test image', im_i, ", Min,Max: %f %f" % (np.min(pred), np.max(pred))
		if (np.min(pred) == np.max(pred)) and (np.min(pred) == 0):
			print 'no feature captured'
			break

		# debug
		if FLAGS.DEBUG:
			print 'pred shape', (pred.shape)
		pred = pred[0, 0, :, :].reshape((h, w)) #pred[0,0,:,:]
		pred_255 = pred * 255  #
		if FLAGS.DEBUG:
			print 'model output255,reshape:',pred_255

		pred_int = np.uint8(pred_255)
		pred_thresh = pred_int.copy()
		ceiling_to_pixel_value = 255
		idx = pred_int[:, :] > thresh_res
		pred_thresh[idx] = ceiling_to_pixel_value
		#add_color = tools.add_colorOverlay(input_resized, pred_int)

		step_evaluate = False
		if step_evaluate:

			k = cv2.waitKey(30) & 0xFF
			while True:
				if k == ord('n'):
					print 'add to fail_list'
					res_fail_list.append(im_i)
					# res_fail_list.append(files[i])
					break
				elif k == ord('y'):
					print 'add to pass_list'
					res_pass_list.append(im_i)
					save_imgs = True
					if save_imgs:
						cv2.imwrite('../classified/MA_k/pass_cv_' + str(im_i) + '.png', image_view)
					break
				elif k == ord('q'):  # ESC
					break
				else:
					k = cv2.waitKey(30) & 0xFF
					if k != 255:
						print 'k:', k  # 81-l, 83-r, 82-u, 84-d

		# debug
		# cv2.imwrite('../testbench/frame_res_tmp.jpg', np.uint8(res))
		if FLAGS.INFO_0:
			print '# show segment result for frame', im_i


	#############################################################################
	# tmp_time0=time.time() #time for a loop 5.88762402534,5.81136989594 start from here
	# CLASSIFICATION

	# IMAGE
	# cv2.imwrite('../testbench/frame_crop_color.jpg',frame_crop_color)
	fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg = ROI(pred_thresh, im_crop_color, h, w, im_i,
																		 save_file=False)
	#	fr_add_cont, new_mask, pre_tensor, r1, r2, c1, c2, screen_out = ROI(pred_thresh, im_crop_color, h, w, im_i,
	#																	 save_file=False)
	
	# time for ROI 5.74937677383
	# tmp_time0 = time.time() #time for a loop 0.0359718799591 start from here
	# print 'time for write',tmp_time2-tmp_time1 # 0.00110197067261, 7.58029007912(no write), 7.33648395538(use write)

	return pred_int, pred_thresh, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg


def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

def EVALUATE_VIDEO_seg_and_classify(bg_model,VIDEO_FILE, num_class,in_ch,show_step=False,save_class_imgs=False,stop=False):  # (v)
		
	n_classes = num_class
	bg_model = FLAGS.seg_model_search_p + bg_model
	video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
	
	video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
	# hy: propID=1 means 0-based index of the frame to be decoded/captured next
	
	if not video.isOpened():
		print "cannot find or open video file"
		exit(-1)
	
	eva_count = 0
	video_frame_i = 0
	confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
	confMat2_TEST = np.zeros((2, 2), dtype=np.float)
	while True and not stop:  # and video_frame_i < 850:
		
		ret, frame = video.read()
		
		if cv2.waitKey(1) & 0xFF == ord('q'):  # hy:press key-q to quit
			print 'key interrupt,press q again to quit completely'
			stop = True
			break
		if ret:  # time for a loop 7.28790903091 start from here
			h,w = frame.shape[0],frame.shape[1]
			video_frame_i += 1
		# print '\n\n########################################### start, frame', video_frame_i
		# print 'frame shape h,w:', h, w  # 1536 2304

			if video_frame_i % 10 == 0: #' and video_frame_i > 3750:#> 1350 and video_frame_i < 1470:
				# setting for white bg frames
				# 1-64:rechts, 81-: links, 510-960:vorn, 990-1320:hinten, 1350-1470:oben, 1500-2550:unten

				#setting for new clear: 2370-2590:vorn, 2600-2700:oben  2700-2980:links 3010-3030:oben, 3030-3070:mv
				#3080-3100:hinten  3600-3730:no obj, 3740-3900,3990-4190:unten

				# setting for white bg frames
				#1-64:rechts,510-960:vorn, 990-1320:hinten, 1350-1470:oben, 1500-2550:unten
				eva_count += 1
				# time for a loop 7.43529987335,7.09782910347 variously, start from here
				use_focus_window = 0
				if use_focus_window:
					crop_x1 = 450  # 550
					crop_y1 = 600  # 700# 300
					area_step_size = 1080  # 740# 640
					crop_x2 = crop_x1 + area_step_size * 1
					crop_y2 = crop_y1 + area_step_size * 1 * settings.h_resize / settings.w_resize
					frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
					frame_crop_color = frame_crop.copy()
				# cv2.imwrite('../testbench/frame_color_tmp.jpg', np.uint8(frame_crop_color))
				else:
					crop_x1 = 0
					crop_y1 = 0

					crop_x2 = 0 + w  # 2300  #1920
					crop_y2 = 0 + h  # 1536  #1080
					frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
					frame_crop_color = frame_crop.copy()

				# debug
				# print "shape:y1,y2,x1,x2:", crop_y1,", ", crop_y2,", ", crop_x1,", ", crop_x2
				# cv2.imshow("frame_cropped", frame_crop)
				# time for a loop 7.08207583427 from here

				# frame_crop = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=320)
				# debug
				# print 'crop size', frame_crop.shape
				# cv2.imshow("TensorFlow Window", imutils.resize(frame_crop.astype(np.uint8), 480))
				# load seg model
				################################################################################################################
				set_image_dim_ordering(dim_ordering='th')  #
				model = load_model(bg_model)
				print 'loaded model', bg_model
				pred_int,pred_thresh,fr_add_cont,old_mask,old_roi_res,new_mask,new_roi_res,\
				   r1, r2, c1, c2, screen_out,roi_whitebg \
					 = do_segment_video(model, frame_crop_color, video_frame_i, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE,in_ch,show_bbox=True)

				################################################################################################################
				#r1, r2, c1, c2 = r1+border, r2-border, c1+border, c2-border
				#frame_crop_roi = frame_crop_roi[r1:r2, c1:c2] #if reduce border
				# time elapsed for classification 0.0430190563202
				######################### get classified result start#################################

				# load classifier model
				new_graph = tf.Graph()

				with tf.Session(graph=new_graph) as sess2:
					ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../logs/")
					saver = tf.train.import_meta_graph(classifier_model)  # (v)

					ckpt.model_checkpoint_path = classifier_model[:-5]

					if ckpt and ckpt.model_checkpoint_path:
						saver = tf.train.Saver()
						saver.restore(sess2, ckpt.model_checkpoint_path)
						print "Evaluation with model", ckpt.model_checkpoint_path
					else:
						print 'not found model'

					tensorImgIn, test_labels, pre_tensor = get_tensor(video_frame_i, new_roi_res, n_classes, cvtcolor=True, screen_out=True)

					target = tools.get_ground_truth_label(video_frame_i, default=False)
					#VIDEO
					RES,demo_frame,stat = get_classify_result(sess2, tensorImgIn, test_labels, video_frame_i, frame, frame_crop,
											  r1, r2, c1, c2,border, screen_out=True,fname='',target=target)


					confMat1_TEST, confMat2_TEST = tools.process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop,
																	 SAVE_CorrectClassified, SAVE_Misclassified, video_frame_i,
																	 target=target)
					print '\neva_count', eva_count
					do_statistics(confMat1_TEST, confMat2_TEST, eva_count)

					#####################################################
					if not show_step:
						if save_class_imgs:
							print 'save path:', FLAGS.save_res_path + 'frame_' + str(video_frame_i)
							cv2.imwrite(FLAGS.save_res_path + 'frame_' + str(video_frame_i) + '.png', pre_tensor)
					else:
						k = cv2.waitKey(30) & 0xFF
						while True and not stop:
							if k == ord('n'):
								print 'add to fail_list'
								#res_fail_list.append(read_path_im + os.path.basename(files_im[i]))
								# res_fail_list.append(files[i])
								break
							elif k == ord('y'):
								print 'add to pass_list'
								save_ori_frame = True
								save_seg_imgs = True
								if save_seg_imgs:
									if save_ori_frame:
										#im_save = cv2.resize(frame_crop_color, (1920, 1080), interpolation=cv2.INTER_CUBIC)  # upsampling
										im_save = frame_crop_color  # upsampling
										#or save image_crop_roi
									else:
										im_save = cv2.resize(frame_crop_color, (h, w))
									print 'save path:',FLAGS.save_res_path + 'frame_'+str(video_frame_i)
									cv2.imwrite(FLAGS.save_res_path +'unten_clearN_' + str(video_frame_i) + '.jpg', im_save)
								# misc.imsave('../classified/MA_1/pass_misc_' + str(im_i) + '.png', im_save)
								break

							elif k == ord('q'):  # ESC
								break
							else:
								k = cv2.waitKey(30) & 0xFF
								if k != 255:
									print 'k:', k  # 81-l, 83-r, 82-u, 84-d

						if cv2.waitKey(1) & 0xFF == ord('q'):
							stop = True
							print 'key interrupt'
							break

					cv2.waitKey(10)  # required for roi_seg

					# tmp_time3=time.time()
					# print 'time for a loop',tmp_time3-tmp_time0 #7.11944699287

		else:
			print 'video end'
			stop = True
	
	stop = True
	return stop



#todo add classification
def EVALUATE_WEBCAM_seg_and_classify(camera_port,bg_model, save_res_path,stop, H,W,in_ch,num_class):  # (cam)
	res_pass_list, res_fail_list, list_of_imgs_res = [], [], []
	dices, dice_l, dice_r, dice_o, dice_u, dice_v, dice_h, dices_cad, count_test, n_classes = 0, 0, 0, 0, 0, 0, 0, 0, 0, 6
	max_dice, min_dice, avg_dice_h, avg_dice_v, avg_dice_r, avg_dice_l, avg_dice_o, avg_dice_u, overall_acc = 0, 1, 0, 0, 0, 0, 0, 0, 0

	confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
	confMat2_TEST = np.zeros((2, 2), dtype=np.float)
	# hy: check camera availability
	camera = cv2.VideoCapture(camera_port)

	# resolution can be set here, up to max resolution will be taken
	camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 2300)
	camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1920)

	# area_step_size_webcam = 1080  # 479 #200

	confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)
	confMat2_TEST = np.zeros((2, 2), dtype=np.float)

	frame_i = 0
	eva_count = 0
	# hy: initialize confmatrix
	confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)

	# if stop == False:
	# if ckpt and ckpt.model_checkpoint_path:
	# Camera 0 is the integrated web cam

	# Number of frames to throw away while the camera adjusts to light levels
	#  ramp_frames = 1

	while True and not stop:  # hy: confirm camera is available
		# Now we can initialize the camera capture object with the cv2.VideoCapture class.
		# All it needs is the index to a camera port.
		print 'Getting image...'

		ret, frame = camera.read()

		# Captures a single image from the camera and returns it in PIL format

		# ret = camera.set(3, 320) #hy use properties 3 and 4 to set frame resolution. 3- w, 4- h
		# ret = camera.set(4, 240)

		#cv2.waitKey(1)
		# A nice feature of the imwrite method is that it will automatically choose the
		# correct format based on the file extension you provide.
		# cv2.imwrite(file, camera_capture)

		####################################  /////////////////////////////


		if frame is not None:
			# print 'frame from webcam obtained'
			# while frame is not None:
			frame_i += 1
			# hy:
			h_frame = frame.shape[0]
			w_frame = frame.shape[1]  # hy: h 1536 x w 2304

			if FLAGS.DEBUG:
				print "h and w", h_frame, ",", w_frame

			if frame_i % 10 == 0:
				eva_count += 1
				use_focus_window = 0
				if use_focus_window:
					crop_x1 = 450  # 550
					crop_y1 = 600  # 700# 300
					area_step_size = 1080  # 740# 640
					crop_x2 = crop_x1 + area_step_size * 1
					crop_y2 = crop_y1 + area_step_size * 1 * settings.h_resize / settings.w_resize
					frame_in = frame[crop_y1:crop_y2, crop_x1:crop_x2]
					frame_in_color = frame_in.copy()
				# cv2.imwrite('../testbench/frame_color_tmp.jpg', np.uint8(frame_crop_color))
				else:
					crop_x1 = 0
					crop_y1 = 0

					crop_x2 = 0 + w_frame  # 2300  #1920
					crop_y2 = 0 + h_frame  # 1536  #1080
					frame_in = frame[crop_y1:crop_y2, crop_x1:crop_x2]
					frame_crop = frame_in.copy()
					frame_crop_view = frame_in.copy()

				# debug
				# print 'crop size', frame_crop.shape
				################################################################################################################
				set_image_dim_ordering(dim_ordering='th')  #
				model = load_model(bg_model)
				if FLAGS.INFO_0:
					print 'loaded model', bg_model

				#webcam pred_int, pred_thresh, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg
				pred_int, pred_255, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, \
				           r1, r2, c1, c2, screen_out, roi_whitebg = do_segment(model, frame_crop,
																			frame_i, H, W,in_ch,show_bbox=True)

				################################################################################################################
				#########                                        CLASSIFICATION                                        #########
				################################################################################################################
				# load classifier model
				new_graph = tf.Graph()

				with tf.Session(graph=new_graph) as sess2:
					ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../logs/")
					saver = tf.train.import_meta_graph(classifier_model)  # (v)

					ckpt.model_checkpoint_path = classifier_model[:-5]

					if ckpt and ckpt.model_checkpoint_path:
						saver = tf.train.Saver()
						saver.restore(sess2, ckpt.model_checkpoint_path)
						print "Evaluation with model", ckpt.model_checkpoint_path
					else:
						print 'not found model'

					tensorImgIn, test_labels, pre_tensor = get_tensor(frame_i, pre_tensor, n_classes,
																	  cvtcolor=True, screen_out=True)

					target = tools.get_ground_truth_label(frame_i, default=False)
					# webcam
					RES, demo_frame, stat = get_classify_result(sess2, tensorImgIn, test_labels, frame_i, frame,
																frame_crop,
																r1, r2, c1, c2, border, screen_out=True, fname='',
																target=target)

					confMat1_TEST, confMat2_TEST = tools.process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop,
																	 SAVE_CorrectClassified, SAVE_Misclassified,
																	 frame_i,
																	 target=target)
					print '\neva_count', eva_count
					do_statistics(confMat1_TEST, confMat2_TEST, eva_count)

					prefix = get_model_index(bg_model, '\d')
					if FLAGS.DEBUG:
						print 'prefix:', prefix

					found_dig = re.search('\d', os.path.basename(bg_model))
					if found_dig:
						dig = found_dig.start()
						prefix = save_res_path + os.path.basename(bg_model)[dig:-5]

					if create_stacked_imgs_for_paper:
						s_size = (320 / 2, 320 / 2)
						list_of_imgs = [cv2.resize(frame_crop_view, (320 / 2, 320 / 2)), cv2.resize(pred_int, s_size),
										cv2.resize(old_mask, (320 / 2, 320 / 2)), cv2.resize(fr_add_cont, s_size),
										cv2.resize(pre_tensor, s_size)]
						winname = 'in-pred-oldmask-newmask-cont-pretensor'
						fn = 'webcam_'+ str(frame_i)
						stacked_imgs = demo_stacked_n_col_images(prefix, fn, list_of_imgs, winname, save_im=False)
						list_of_imgs_res.append(stacked_imgs)

					#####################################################
					if not step_show:
						pass
						#print 'save path:', FLAGS.save_res_path + 'frame_' + str(frame_i)
						#cv2.imwrite(FLAGS.save_res_path + 'frame_' + str(frame_i) + '.png', pre_tensor)
					elif not FLAGS.CLOSE_ALL:
						k = cv2.waitKey(30) & 0xFF
						while True and not stop:
							if k == ord('n'):
								print 'add to fail_list'
								# res_fail_list.append(read_path_im + os.path.basename(files_im[i]))
								# res_fail_list.append(files[i])

								break
							elif k == ord('y'):
								print 'add to pass_list'
								save_ori_frame = True
								save_seg_imgs = True
								if save_seg_imgs:
									if save_ori_frame:
										# im_save = cv2.resize(frame_crop_color, (1920, 1080), interpolation=cv2.INTER_CUBIC)  # upsampling
										im_save = frame_crop  # upsampling
									# or save image_crop_roi
									else:
										img_h, img_w = 320,320
										im_save = cv2.resize(frame_crop, (img_h, img_w))
									print 'save path:', FLAGS.save_res_path + 'frame_' + str(frame_i)
									cv2.imwrite(FLAGS.save_res_path + 'unten_clearN_' + str(frame_i) + '.jpg',
												im_save)
								# misc.imsave('../classified/MA_1/pass_misc_' + str(im_i) + '.png', im_save)
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

					cv2.waitKey(10)  # required for roi_seg


		print 'no frame retrieved'

	del (camera)
	return stop


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


def demo_final_seg_result(fn, res, fr_add_cont, image_crop_roi, prefix, save_im=False,save_stack_imgs=False):  # 2
	if not FLAGS.CLOSE_ALL:
		r_gray1, c_gray1 = res.shape
		image_crop_roi = cv2.resize(image_crop_roi, (FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE))
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
		


def demo_stacked_n_col_images(prefix,fn, list_of_imgs, winname, save_im=False):  # 2
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
	c_comb = c_comb+(width-1)*frame_border
	comb_im = np.zeros(shape=(r_comb, c_comb, dim_im), dtype=np.uint8)
	white = np.zeros(shape=(r_comb,frame_border,dim_im),dtype=np.uint8)
	white2 = np.zeros(shape=(r_comb,frame_border),dtype=np.uint8)
	white.fill(255)
	white2.fill(255)
	
	current_column = 0
	for im in list_of_imgs:
		if len(im.shape) == 3:
			comb_im[:(im.shape[0]), current_column:current_column+im.shape[1]] = im
			if current_column+im.shape[1] < c_comb:
				comb_im[:(im.shape[0]),current_column+im.shape[1]:current_column+im.shape[1]+frame_border] = white
		else:
			comb_im[:(im.shape[0]), current_column:current_column+im.shape[1]] = im[:,:,None]
			if current_column+im.shape[1] < c_comb:
				comb_im[:(im.shape[0]),current_column+im.shape[1]:current_column+im.shape[1]+frame_border] = white2[:,:,None]
	
		current_column = current_column + im.shape[1] + frame_border

	if not FLAGS.CLOSE_ALL and not do_multiple_test:
		cv2.imshow(winname,comb_im)
		cv2.waitKey(5)
	if  save_im:
		cv2.imwrite(prefix + 'comb_' + fn + '.png', comb_im)
	print 'comb im shape:', comb_im.shape
	return comb_im

def demo_stacked_n_row_images(prefix,fn,list_of_imgs_res,winname,save_im=False):
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
	
	cv2.imshow(winname,comb_im)
	if save_im:
		cv2.imwrite(prefix + fn + '.png', comb_im)


def EVA_IMAGE_classify_pose(read_path_im,
                                 frame,frame_crop,frame_crop_roi,
                                 crop_y1, crop_y2, crop_x1, crop_x2,
                                 count_test,confMat1_TEST,confMat2_TEST,border,file_i,n_classes,fname=''):
	
	# load classifier model
	new_graph = tf.Graph()
	#print 'frame ori size:',frame_crop.shape   #(1080, 1920, 3)
	#print 'frame crop size:',frame_crop_roi.shape   #(199, 129, 3)
	
	with tf.Session(graph=new_graph) as sess2:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../logs/")
		saver = tf.train.import_meta_graph(classifier_model)  # (v)
		
		ckpt.model_checkpoint_path = classifier_model[:-5]
		
		if ckpt and ckpt.model_checkpoint_path:
			saver = tf.train.Saver()
			saver.restore(sess2, ckpt.model_checkpoint_path)
			if not result_for_table:
				print "Evaluation with model", ckpt.model_checkpoint_path
		else:
			print 'not found model'
		
		
		tensorImgIn, test_labels, pre_tensor = get_tensor(file_i, frame_crop_roi, n_classes, cvtcolor=True, screen_out=True)
		
		target = tools.get_ground_truth_label_im(read_path_im, default=False)
		#target = tools.get_ground_truth_label(file_i, default=False) #for video, by frame number

		#IMAGE
		RES,demo_im,stat = get_classify_result(sess2, tensorImgIn, test_labels, file_i, frame, frame_crop,
		                          crop_y1, crop_y2, crop_x1, crop_x2,border,screen_out=True,fname=fname,target=target)
		
		
		confMat1_TEST, confMat2_TEST = tools.process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop,
		                                                 SAVE_CorrectClassified, SAVE_Misclassified, file_i,
		                                                 target=target)

		overall_acc = do_statistics(confMat1_TEST, confMat2_TEST, count_test)
		
		cv2.waitKey(10)  # required for roi_seg
		return pre_tensor, overall_acc,demo_im,confMat1_TEST,stat
	
create_stacked_imgs_for_paper = True
#MA
def EVA_IMAGE_seg_and_classify(MODEL_ID,bg_model,h, w,
                                    in_ch=1,best_avg=0,save_res_path='',save_imgs=False,file_i=1,
with_gt=False,step_show=False,stop=False,test_data_path=''):
	print 'Model ID:', MODEL_ID
	res_pass_list, res_fail_list, list_of_imgs_res = [], [], []
	dices, dice_l,dice_r,dice_o,dice_u,dice_v,dice_h, dices_cad, count_test, n_classes = 0,0,0,0,0,0,0, 0, 0, 6
	max_dice,min_dice,avg_dice_h,avg_dice_v,avg_dice_r,avg_dice_l,avg_dice_o,avg_dice_u, overall_acc = 0,1, 0,0,0,0,0,0,0
	
	confMat1_TEST = np.zeros((n_classes, n_classes), dtype=np.float)  # hy collect detailed confusion matrix
	confMat2_TEST = np.zeros((2, 2), dtype=np.float)
	
	if not with_gt:
		read_paths_im, files_im = get_read_path_and_files_for_im(read_from_file=False,data_path=test_data_path)
		files_m, read_paths_m = files_im, read_paths_im
	elif do_6face_classification:
		read_paths_im, read_paths_m, files_im, files_m = get_read_path_and_files_for_im_m_6cl(read_from_file=False)
	else:
		read_paths_im, read_paths_m, files_im, files_m = get_read_path_and_files_for_im_m(read_from_file=False,data_path=test_data_path)

	# load seg model
	set_image_dim_ordering(dim_ordering='th')  #
	#summary:cambin np.int32(mask > 0 is used
	#start from cambin_45 show positive result with 0.35loss,good result on clear images
	#at cambin_106 with 0.13loss, on clear images although no detected res image there is very clear feature like fire on corner, edges, glass, wheels
	#at cambin_137 with 0.0917loss no big change

	model = load_model(bg_model)
	if FLAGS.INFO_0:
		print 'loading model', bg_model

	# debug
	# model.summary()
	
	if result_for_table:
		print 'True/False', 'No.', 'Name', 'TargetLabel', 'PredictLabel', 'Precision', \
			'Top2', 'Top2_pres', 'Top3', 'Top3_pres', 'Top4', 'Top4_pres', 'Top5', 'Top5_pres', 'last', 'last_pres'
	
	cust_list=['vorn_mac_im_1640','vorn_mac_im_200','hinten_mac_im_1840',
	          'links_mac_im_1100','rechts_2_mac_im_4360','unten_2_mac_im_7380',
	          'unten_2_mac_im_7660','oben_mac_im_5280','oben_mac_im_5720']
# if
	
	for im_ori, ref_mask, i, read_path_im,read_path_m in zip(files_im, files_m, xrange(len(files_im)) ,read_paths_im,read_paths_m ):
		fn = os.path.basename(im_ori)[:-4]
		if i >-1:#> 46 and i < 54:
		#if fn in cust_list: # 0 and i < 4:#> 46 and i < 54:
			if FLAGS.INFO_0:
				print 'read path:',read_path_im
			
			im_ori_view = cv2.imread(read_path_im + im_ori)
			im_crop = im_ori_view.copy()
			im_ori_view = cv2.resize(im_ori_view, (h, w))
			
	  #IMAGE
		#pred_int, pred_thresh, fr_add_cont, old_mask, old_roi_res, new_mask, pre_tensor, r1, r2, c1, c2, screen_out,roi_whitebg
			pred_int,pred_thresh, \
			fr_add_cont, old_mask, \
			old_roi_res, \
			     new_mask, image_crop_roi, r1, r2, c1, c2, screen_out,roi_whitebg \
				= do_segment_im(model, im_ori_view, im_crop, i, h, w,in_ch,show=False,save=False)
			
			print 'screen out:', screen_out
			pred_thresh = new_mask.copy() #use optimized mask for dice calc
		
			if len(pred_thresh.shape) == 3:
				pred_thresh = cv2.cvtColor(pred_thresh,cv2.COLOR_RGB2GRAY)
	
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
					print '\nDice for',fn,':',dice
					dices += dice
					if dice < min_dice:
						min_dice = dice
					if dice > max_dice:
						max_dice = dice
					if do_6face_classification:
						num_imgs_of_class = len(os.listdir(read_path_im))
						if 'vorn' in read_path_im:
							dice_v += dice
							avg_dice_v = dice_v/num_imgs_of_class
						if 'hinten' in read_path_im:
							dice_h += dice
							avg_dice_h = dice_h /num_imgs_of_class
						if 'links' in read_path_im:
							dice_l += dice
							avg_dice_l = dice_l /num_imgs_of_class
						if 'rechts' in read_path_im:
							dice_r += dice
							avg_dice_r = dice_r /num_imgs_of_class
						if 'oben' in read_path_im:
							dice_o += dice
							avg_dice_o = dice_o /num_imgs_of_class
						if 'unten' in read_path_im:
							dice_u += dice
							avg_dice_u = dice_u /num_imgs_of_class
	
			#screen_out = True
			#if save_imgs and screen_out:
			
			prefix = get_model_index(bg_model,'\d')
			if screen_out:
				if FLAGS.DEBUG:
					print 'prefix:',prefix
				found_dig = re.search('\d', os.path.basename(bg_model))
				if found_dig:
					dig = found_dig.start()
					prefix = save_res_path + os.path.basename(bg_model)[dig:-5]
				else:
					print 'no file found'
				
				reduce_border = False
				if reduce_border:
					#image_crop_roi = cv2.resize(im_crop,(h,w))[r1+border:r2-border, c1-border:c2+border] #crop_y1:crop_y2, crop_x1:crop_x2
					image_crop_roi = cv2.resize(im_crop,(h,w))[r1+border:r2-border, c1+border:c2-border] #crop_y1:crop_y2, crop_x1:crop_x2
				if FLAGS.INFO_0:
					print 'image_crop_roi shape (tf ordering hwc):',image_crop_roi.shape #tf ordering channel last
				
				
				if min(image_crop_roi.shape) > 1:
					#demo_final_seg_result(fn,pred_thresh,fr_add_cont,image_crop_roi,prefix,screen_out,save_im=False,save_stack_imgs=False)
					if do_6face_classification:
						pre_tensor_view, overall_acc,demo_im,confMat1_TEST,stat \
							= EVA_IMAGE_classify_pose(read_path_im,im_ori_view,im_crop,image_crop_roi,
					                             r1, r2, c1, c2,count_test,confMat1_TEST,confMat2_TEST,border,i,n_classes,fn)
				else:
					print 'tensor size too small'
					cv2.putText(fr_add_cont, 'too small ROI', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
					            color=(0, 255, 0), thickness=2)
					pre_tensor_view = np.zeros((h,w,3),np.uint8)
			else:
				print 'no ROI'
				cv2.putText(fr_add_cont, 'no ROI', org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
				            color=(0, 255, 0), thickness=2)
				pre_tensor_view = np.zeros((h,w,3),np.uint8)
			
			if create_stacked_imgs_for_paper:
				s_size = (320/2,320/2)
				if read_path_im == read_path_m:
					list_of_imgs = [cv2.resize(im_ori_view, s_size), cv2.resize(pred_int, s_size),
					                cv2.resize(old_mask, s_size), cv2.resize(new_mask, s_size), cv2.resize(fr_add_cont, s_size),
					                cv2.resize(roi_whitebg, s_size), cv2.resize(pre_tensor_view, s_size)]
					#,cv2.resize(stat,s_size)
					#list_of_imgs = [cv2.resize(im_ori_view, (320/2,320/2)),cv2.resize(pred_int, s_size),
					 #               cv2.resize(old_mask, (320/2,320/2)),cv2.resize(fr_add_cont, s_size)]
					
          			#cv2.resize(roi_whitebg, (320/2,320/2)),cv2.resize(pre_tensor_view, (320/2,320/2))
					
					winname = 'in-pred-oldmask-newmask-cont-pretensor'
				else:
					#list_of_imgs = [cv2.resize(im_ori_view,s_size),cv2.resize(ref_mask_thresh,s_size),cv2.resize(pred_int,s_size),
				  #              cv2.resize(old_mask, s_size),cv2.resize(new_mask,s_size),cv2.resize(fr_add_cont,s_size),
				  #              cv2.resize(roi_whitebg,s_size),cv2.resize(pre_tensor_view,s_size),cv2.resize(stat,s_size)]
					
					
					#list_of_imgs = [cv2.resize(im_ori_view, (320 / 2, 320 / 2)), cv2.resize(pred_int, s_size),
					 #               cv2.resize(old_mask, (320 / 2, 320 / 2)), cv2.resize(fr_add_cont, s_size)]
					
					list_of_imgs = [cv2.resize(im_ori_view, s_size),
					                cv2.resize(new_mask, s_size), 					                cv2.resize(roi_whitebg, s_size)]
					
					winname = 'in-newmask-whbg'
				stacked_imgs = demo_stacked_n_col_images(prefix,fn, list_of_imgs, winname, save_im=False)
				list_of_imgs_res.append(stacked_imgs)
			#create_stacked_imgs_for_paper = True
		#if do_multiple_test:#
		#cv2.imwrite(SAVE_MUL_RES_PATH + MODEL_ID + '_p_tensor_' + os.path.basename(files_im[i]), pre_tensor_view) #SAVE setting
		#cv2.imwrite(FLAGS.save_res_path +'p_tens_'+ os.path.basename(files_im[i]),pre_tensor_view)

		get_series_result = True
		#if with_gt and show:
		#cv2.waitKey()
		if step_show and not FLAGS.CLOSE_ALL and not stop:
			k = cv2.waitKey(30) & 0xFF
			while True and not stop:
				if k == ord('n'):
					print 'add to fail_list:',fn
					res_fail_list.append(read_path_im + os.path.basename(files_im[i]))
					#res_fail_list.append(files[i])
					break
				elif k == ord('y'):
					print 'add to pass_list:',fn
					res_pass_list.append(read_path_im + os.path.basename(files_im[i]))
					save_imgs = True
					if save_imgs:
						#cv2.imwrite(FLAGS.save_res_path +'black'+ os.path.basename(files_im[i]),black)
						if do_6face_classification:
							pass
							#cv2.imwrite(FLAGS.save_res_path +'demo_im_'+ os.path.basename(files_im[i]),demo_im)
						#cv2.imwrite(FLAGS.save_res_path +'roi_res_'+ os.path.basename(files_im[i]),image_crop_roi)
						
						#cv2.imwrite(FLAGS.save_res_path +'pre_tensor_'+ os.path.basename(files_im[i]),pre_tensor_view)
						cv2.imwrite(FLAGS.save_res_path +'stack_'+ os.path.basename(files_im[i]),stacked_imgs)
					break

				elif k == ord('q'): #ESC
					stop = True
					break
				else:
					k = cv2.waitKey(30) & 0xFF
					if k != 255:
						print 'k:',k #81-l, 83-r, 82-u, 84-d


			if cv2.waitKey(1) & 0xFF == ord('q'):
				print 'key interrupt'
				break

	if not with_gt:
		print bg_model,'test package:', read_path_im
		print 'res_fail_list=',res_fail_list,  '\nres_pass_list=',res_pass_list
		print 'num of fail:',len(res_fail_list), '\nnum of pass:',len(res_pass_list)

	def save_to_file(file,lines):
		lines = '\n'.join(lines)
		with open(file,'w') as f:
			f.writelines(lines)

	save_eva_to_file = False
	if save_eva_to_file:
		keyword = os.path.basename(os.path.normpath(read_path_im))
		save_to_file('../Test_Images/img_list/fail_' + keyword + '_' + os.path.basename(bg_model)[:-5]+'.txt',res_fail_list)
		save_to_file('../Test_Images/img_list/pass_' + keyword + '_' + os.path.basename(bg_model)[:-5]+'.txt',res_pass_list)
	cv2.destroyAllWindows()

	if count_test > 0:
		avg_dice = float(dices / count_test)
	else:
		avg_dice = 0
	if avg_dice > best_avg:
		best_avg = avg_dice
	print '\nseg avg dice:', avg_dice, 'max dice:',max_dice,', min dice:', min_dice, ', ',bg_model
	if do_6face_classification:
		print '6classifier overall_acc', overall_acc
		print tools.print_label_title()
		print confMat1_TEST
		print 'seg avg_v:', avg_dice_v,  'seg avg_h:',avg_dice_h, ', seg avg_l:', avg_dice_l, ', \nseg avg_r:', avg_dice_r \
			, ', seg avg_u:', avg_dice_u, ', seg avg_o:', avg_dice_o
	else:
		overall_acc = 0
	print 'classifier model', classifier_model
	print 'use_cut:',use_cut, ', use_limit:', use_limit
	
	
	num_total_ims = len(list_of_imgs_res)
	print 'list imgs res:', num_total_ims
	#create_stacked_imgs_for_paper = True
	if create_stacked_imgs_for_paper and num_total_ims>0 and num_total_ims<10:
		if not do_multiple_test:
			MODEL_ID = ''
		demo_stacked_n_row_images(prefix,MODEL_ID,list_of_imgs_res,winname,save_im=True)
		
	return best_avg, bg_model,overall_acc,stop


	# return image_crop_roi,screen_out

##########################################################################################################
if EVALUATE_VIDEO == 1:
	
	TEST_VIDEO_FULL = True
	if TEST_VIDEO_FULL:
		print '####################################'
		print 'Starting evaluation with k model'
		print Seg_MODEL_to_load, classifier_model
		video_list = ['full/']
		TestFace = 'full'
		video_window_scale = 1
		VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
		print VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label
		stop = False
		while not stop:
			stop = EVALUATE_VIDEO_seg_and_classify(Seg_MODEL_to_load,VIDEO_FILE, num_class=6,in_ch=1,show_step=False,save_class_imgs=False,stop=False)
	
	else:
		video_list = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
		
		for video_index in xrange(len(video_list)):
			TestFace = video_list[video_index][:-1]  # all # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
			print 'Test face:', TestFace
			
			# TestFace = settings.LABELS[video_index][:-1] #'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
			VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, scale=2)
			stop = False
			while not stop:
				stop = EVALUATE_VIDEO_seg_and_classify(Seg_MODEL_to_load,VIDEO_FILE, 6,in_ch=1,show_step=False,save_class_imgs=False,stop=False)
			print 'test face:', TestFace, 'done\n'
	

if EVALUATE_WEBCAM:
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../testbench/6classes/")
		saver = tf.train.import_meta_graph(classifier_model)  # (web)

		ckpt.model_checkpoint_path = classifier_model[:-5]
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Evaluation live frames with", ckpt.model_checkpoint_path
		else:
			print 'not found model'
		print 'Test with Webcam starting ...'
		# Camera 0 is the integrated web cam on my netbook
		camera_port = 0
		# EVALUATE_WITH_WEBCAM_track_roi(camera_port,n_classes)
		bg_models = sorted([s for s in os.listdir(FLAGS.seg_model_search_p) if Seg_MODEL_to_load in s])
		bg_model = bg_models[-1] if len(bg_models) > 1 else bg_models[0]
		print 'bg model:', bg_model
		save_res_path = PROJ_DIR + 'testbench/k_imgs/'  # /seg_mul_imgs/'
		bg_model = FLAGS.seg_model_search_p + bg_model
		stop = False
		while not stop:
			stop = EVALUATE_WEBCAM_seg_and_classify(camera_port,bg_model, save_res_path,False, FLAGS.IMAGE_SIZE,FLAGS.IMAGE_SIZE,in_ch, num_class=6) #camera_port, stop, H,W,in_ch,num_class

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()


if EVA_IMAGE_seg_and_class == 1:
	bg_models = sorted([s for s in os.listdir(FLAGS.seg_model_search_p)
	                    if Seg_MODEL_to_load in s]) # if 'real_test_06-0.13pt3.hdf5' in s])
	#6cl_only_profile_test_03-0.21_keep0.63
	bg_model = bg_models[-1] if len(bg_models) > 1 else bg_models[0]
	print 'bg model:', bg_model
	bg_model = FLAGS.seg_model_search_p + bg_model

	best_avg = 0
        test_data_path = '/home/hy/unet/Data/'
        

	seg_avg, model_name, cl_overall_acc,stop = EVA_IMAGE_seg_and_classify\
		(MODEL_ID,bg_model, FLAGS.IMAGE_SIZE,FLAGS.IMAGE_SIZE,in_ch,best_avg, save_res_path=FLAGS.save_res_path,save_imgs=True,
		 file_i=1, with_gt=FLAGS.with_gt,step_show=step_show,stop=False,test_data_path=test_data_path)



