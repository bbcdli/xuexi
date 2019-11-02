# change logs are located in tensor_train.py
import tensorflow as tf
import Image
import cv2
import numpy as np
import sys
from keras.backend import set_image_dim_ordering
from keras.models import load_model
import keras
import ImageDraw
import ImageFilter
import ImageOps
import time
from functools import wraps
from random import randint
import os
import datetime
import settings  # hy: collection of global variables
import tools
import time
import prep_image
from sklearn import datasets
import math
import imutils
from PIL import Image  # hy: create video with images
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import seg_net_arch as u_a

# from background_learning_s import dice_coef_loss
# tf.python.control_flow_ops = tf #hy:for remote
# KERAS_BACKEND=tensorflow python -c "from keras import backend"
# Using TensorFlow backend.

#####################################################################################################
# activate global var
settings.set_global()
start_time = time.time()

EVALUATE_IMAGES = 0
EVALUATE_VIDEO = 1
generate_tr_data = False

# bg_model = 'weights208_0.0254.hdf5'# very fewer area detected than the previous model with low loss
bg_model = 'weights_no_full_433_696_p05_offi.hdf5'
# bg_model = 'weights41_0.11_offi.hdf5'
# bg_model = 'weights.05_second_good.hdf5'
# bg_model = 'weights216_normal1.hdf5'
# bg_model = 'weights41_0.11.hdf5'
# bg_model = 'weights27_0.3.hdf5'

classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6all_0_tr_0.88-111.meta"  #
# classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6all_6_0.6-341_part2.meta"  #
# classifier_model = "../logs/" + "model_GD360_h184_w184_c6all_7_0.72-171_rep_part2_addSeg.meta"
# classifier_model = "../testbench/6classes/" + "model_GD360_h184_w184_c6_3conv_O1.0_U1.0_V1.0_4_0.8-781.meta"
n_hidden = 360  # n_hidden = classifier_model.split('conv')[1][0:3]
do_active_fields_test = 0
dropout = [0.3, 0.3, 0.5, 0.5]  # 3,4,5,5
# dropout = [0.3]  # 3,4,5,5
dropout_1s = [1] * len(dropout)

##########

TEST_CONV_OUTPUT = False
result_for_table = 1
SAVE_Misclassified = 0
SAVE_CorrectClassified = 0

###########

LABEL_LIST = '../FileList.txt'
LABEL_PATH = settings.data_label_path

# LABEL_LIST_TEST = '../FileList_TEST1_sing.txt'

LABEL_LIST_TEST = '../FileList_TEST1.txt'
# LABEL_LIST_TEST = '../FileList_TEST.txt'
LABEL_PATH_TEST = settings.test_label_path


######################################################################################################
def demo_result_imgs(file1, file2, file3, frame_i=1, save_file=False, demo=False):
	if save_file:
		cv2.imwrite('../testbench/frame_res_' + bg_model[:-5] + '%03d.jpg' % frame_i, np.uint8(file1))
		cv2.imwrite("../testbench/frame_combined_%03d.jpg" % frame_i, file2)
		cv2.imwrite("../testbench/frame_color_%03d.jpg" % frame_i, file3)
	if demo:
		cv2.namedWindow('seg_feature', cv2.WINDOW_AUTOSIZE)
		# cv2.namedWindow('result_win', cv2.WINDOW_NORMAL)
		cv2.putText(np.uint8(file1), 'frame ' + str(frame_i), org=(320 / 10, 320 / 8),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
		            color=(0, 255, 0), thickness=2)
		cv2.imshow('seg_feature', np.uint8(file1))
	# debug
	# cv2.imshow('combined',file2)
	# cv2.imshow('color',file3)
	# cv2.waitKey(20)


def get_bounding_box(conture, img=None):
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
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.waitKey(10)
	return (x, y, w, h)


def get_roi_with_white_bg_seg(roi, base_img, w, h, factor=1):
	overlay = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi_PIL = Image.fromarray(overlay)
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
	
	print 'datas_obj format', datas_obj.size
	
	# way 1
	for data_obj, data_fr in zip(datas_obj, datas_fr):
		if white(data_obj):
			newData.append((data_fr[0], data_fr[1], data_fr[2]))
		else:
			newData.append((255, 255, 255))
		
		## convert PIL back to CV2
	datas_roi.putdata(newData)
	pil_image = datas_roi.convert('RGB')
	open_cv_image = np.array(pil_image)
	# Convert RGB to BGR
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	
	return open_cv_image


def get_roi_with_white_bg_cut(roi, base_img, w=184, h=184, factor=1):
	overlay = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi_PIL = Image.fromarray(overlay)
	# datas_obj = roi_PIL.getdata()
	
	fr = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
	fr_PIL = Image.fromarray(fr)
	
	new_im = Image.new("RGB", (w, h))  ## default black!
	new_im.paste((255, 255, 255), (0, 0, w, h))  ##
	
	# time for cut 5.85256004333 start from here
	def white2(p_col, p_row):
		counter_point_white_col = (p_col == [255, 255, 255]).sum()
		counter_point_white_row = (p_row == [255, 255, 255]).sum()
		return counter_point_white_col, counter_point_white_row
	
	# print 'datas_obj format', datas_obj.size
	
	# http://effbot.org/zone/pil-pixel-access.htm
	out = Image.new(roi_PIL.mode, roi_PIL.size, None)  # w h
	p_roi = np.asarray(roi)  # h w ch
	
	# print 'p31 shape', p31[0][0][0], 'shp2', p31[0][0][1], 'shp point', p31[0][0]
	
	in_pixel = fr_PIL.load()
	out_pixel = out.load()
	tmp_time0 = time.time()
	'''
 for row in xrange(h):
  for col in xrange(w):
   thresh1, thresh2 = white2(p31[row, 0:h - 1], p31[0:w - 1, col])
   if thresh1 > 10 * 1 and thresh2 > 10 * 1:
    out_pixel[col, row] = in_pixel[col, row]
   else:
    out_pixel[col, row] = (255, 255, 255)
 '''
	for col in xrange(w):
		for row in xrange(h):
			thresh1, thresh2 = white2(p_roi[row, 0:h - 1], p_roi[0:w - 1, col])
			if thresh1 > 10 * 11 and thresh2 > 10 * 11:
				out_pixel[col, row] = in_pixel[col, row]
			
			else:
				# out_pixel[col, row] = (0, 0, 0)
				out_pixel[col, row] = (255, 255, 255)
	
	# cv2.waitKey(5900)
	# time for cut 0.00150513648987 start from here
	data_roi = out
	pil_image = data_roi.convert('RGB')
	open_cv_image = np.array(pil_image)
	# Convert RGB to BGR
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	tmp_time3 = time.time()
	print 'time for cut', tmp_time3 - tmp_time0
	
	################
	gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
	ret, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
	###############
	
	contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		largest_areas = sorted(contours, key=cv2.contourArea)
		x, y, box_w, box_h = get_bounding_box(largest_areas[-1], open_cv_image)  # x,y:top-left coord
		if box_w > 30 and box_h > 30:
			screen_out = True
			print 'x, y, w, h', x, y, box_w, box_h
			box_h = int(0.99 * box_h)
			box_w = int(0.99 * box_w)
			open_cv_image = open_cv_image[y:y + box_h, x:x + box_w]  # [crop_y1:crop_y2, crop_x1:crop_x2]
			open_cv_image = cv2.resize(np.uint8(open_cv_image), (h, w))
		else:
			screen_out = False
	else:
		screen_out = False
	return open_cv_image, screen_out


def ROI(obj_area, im_crop, w, h, im_i=0, save_file=False):  # obj_area size 320x320 #(roi)
	# input
	# print 'frame_crop', frame_crop, 'w,h',w,h
	fr = im_crop
	# fr = cv2.imread(frame_crop)
	fr_ori = fr.copy()
	
	# obj_area = cv2.imread(obj_area)
	# gray = cv2.cvtColor(obj_area, cv2.COLOR_BGR2GRAY)
	
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
	
	def find_contour_mul(obj_area):
		contours, screen_out = find_contour(obj_area, 150)
		if not screen_out:
			print 'try with another threshold: 5'
			contours, screen_out = find_contour(obj_area, 15)
			if not screen_out:
				print 'try with another threshold: 15'
				contours, screen_out = find_contour(obj_area, 2)
		return contours, screen_out
	
	#########################################################
	contours, screen_out = find_contour_mul(obj_area)
	
	if screen_out:
		# time for contour 0.000312089920044
		fr_add_cont = cv2.resize(np.uint8(fr), (h, w))  # 0,1
		fr_ori_int8 = cv2.resize(np.uint8(fr_ori), (h, w))  # 0,1
		######################################
		largest_areas = sorted(contours, key=cv2.contourArea)  # 1
		# an operation, returns fr - masked with the largest contour
		cv2.drawContours(fr_add_cont, [largest_areas[-1]], 0, (255, 255, 255, 255), -1)  # 1
		cv2.namedWindow('fr_add_contour', cv2.WINDOW_NORMAL)
		cv2.putText(fr_add_cont, 'frame ' + str(im_i), org=(w / 10, h / 8),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
		            color=(0, 255, 0), thickness=2)
		cv2.imshow('fr_add_contour', fr_add_cont)
		# alternative way: feed the learned segment to classification model
		# way 2 and 3: cut or seg
		# time for cut contour, get res 5.75804781914
		roi_res, screen_out = get_roi_with_white_bg_cut(fr_add_cont, fr_ori_int8, w, h, 1)
	# debug
	# print 'roi_res shape', roi_res.shape
	else:
		print 'no contour found'
		screen_out = False
		roi_res = np.zeros((w, h, 3), np.uint8)  # np.zeros((300, 300, 3), np.uint8
	# roi_res[:] = (0, 0, 255) #fill white
	
	# time for contour,add mask 0.00493407249451 stop here
	# Get bounding box and smaller region from that
	# todo way 1
	# x, y, bw, bh = get_bounding_box(largest_areas[-1],fr_cont)
	
	#####################################################################
	
	# debug
	if save_file:
		cv2.imwrite('../testbench/testImg/tensorInSeg_%03d.jpg' % im_i, roi_res)  # can be used to generate test images
		print 'file saved in ../testbench/testImg/'
	# cv2.waitKey(10)
	return roi_res, screen_out


# ROI('../testbench/frame_res_tmp0.jpg','../testbench/frame_crop_color0.jpg',320,320,1,False)

def get_classify_result(sess, test_image, test_labels, im_i, frame, frame_crop,
                        crop_y1, crop_y2, crop_x1, crop_x2, screen_out=False):
	# print 'frame shape', frame.shape[0], frame.shape[1]
	######################### Tensorflow
	batch_xs, batch_ys = test_image, test_labels
	
	# print 'batch_xs, batch_ys:', batch_xs, ', ', batch_ys
	output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": dropout_1s})
	
	# print("Output for external=",output)
	output = tools.convert_to_confidence(output)  #
	np.set_printoptions(precision=3)
	
	RES = np.argmax(output)  # hy predicted label
	
	label_pred_str = settings.LABELS[RES][:-1]
	# hy: for sub-classes
	# label_pred_str, label_pred_num = tools.convert_result(RES) # hy use it when sub-classes are applied
	# RES_sub_to_face = class_label #hy added
	# print "target, predict =", target, ', ', RES  # hy
	
	
	'''
	if RES == target:
		label2_TEST = 0
		pred2_TEST = 0

		name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % im_i
		tools.SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)  # (v)

	else:
		label2_TEST = 1
		pred2_TEST = 1
		name_str = settings.Misclassified + "/frame_crop%d.jpg" % im_i
		tools.SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)

	confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[
		                                         label2_TEST, pred2_TEST] + 1
	'''
	
	if screen_out:
		cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color=(0, 255, 0), thickness=2)
		prob_str = str(output[0][RES])[:4]  # label index starts from 0
		# prob_str = str(output[0][3])[:4]
		cv2.putText(frame, "predicted: " + label_pred_str + ' acc:' + prob_str,
		            org=(frame.shape[1] / 3, frame.shape[0] / 10),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=4)
		# cv2.putText(frame, label_pred_str, org=(800, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3 )
		
		# print 'video.get',str(video.get(1))
		# cv2.putText(frame, "prob:" + prob_str, org=(w_frame / 10, h_frame / 8),
		#            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
		cv2.putText(frame, 'frame ' + str(im_i), org=(20, 50),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
		
		frame_demo = imutils.resize(frame, width=600)
		
		cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Demo", frame_demo)
	
	return int(RES)


def get_tensor(im_i, pre_tensor, n_classes, cvtcolor, screen_out):
	# tensorImgIn = cv2.imread('../testbench/frame_color_tmp.jpg')
	# transform color and size to fit trained classifier model
	if screen_out:
		cv2.namedWindow('pre_tensor', cv2.WINDOW_NORMAL)
		cv2.putText(pre_tensor, 'frame ' + str(im_i), org=(320 / 10, 320 / 8),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
		            color=(0, 255, 0), thickness=2)
		cv2.imshow('pre_tensor', pre_tensor)
	
	if cvtcolor:
		pre_tensor = cv2.cvtColor(pre_tensor, cv2.COLOR_BGR2GRAY)
	
	# in case gray image as test image, no need to cvt
	
	test_image = cv2.resize(pre_tensor, (settings.h_resize, settings.w_resize))
	test_image = np.asarray(test_image, np.float32)
	
	tensorImgIn = test_image.reshape((-1, test_image.size))
	tensorImgIn = np.expand_dims(np.array(tensorImgIn), 2).astype(np.float32)
	tensorImgIn = tensorImgIn / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
	
	test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
	return tensorImgIn, test_labels


def do_statistics(confMat1, confMat2, data_length):
	# for i in range(0, len(settings.LABELS)):
	#	confMat1[i, :] = confMat1[i, :] / np.sum(confMat1[i, :])
	tools.print_label_title()
	print confMat1
	tp = confMat2[0, 0]
	tn = confMat2[1, 1]
	overall_acc = round(tp / data_length, 2)
	print 'TEST overall acc:', overall_acc
	
	return overall_acc


def do_segment(model, im_crop_color, im_crop, im_i, h, w):
	####### convert into the shape for seg model input
	print w, h
	w = 320
	h = 320
	im_crop = imutils.resize(cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY), width=320)
	im_crop = cv2.resize(im_crop, (h, w))
	im_crop = np.float32(im_crop.reshape(h, w))
	im_crop = im_crop / 255.0
	
	images = np.zeros((3, 1, h, w))  # 3,1,320,320
	images[1, :, :, :] = im_crop
	images = images[1:-1, :, :, :]  # hy: convert to (1, 1, 320, 320) to fit bg model input shape
	# debug
	# print '2-shape of test images', images.shape  # (1, 1, 320, 320)
	
	######################
	mean, stdev = tools.calc_mean_stdev(images)
	
	images_original = images.copy()
	images = (images - mean) / stdev
	
	for i in range(0, images.shape[0]):
		start = time.time()
		result = model.predict(images[i, :, :, :].reshape(1, 1, h, w), batch_size=1)  # old
		end = time.time()
		print 'time elapsed for calc seg feature', (end - start), 's'  # 1.24031400681 s
		print 'Test image', im_i, ", Min,Max: %f %f" % (np.min(result), np.max(result))
		if (np.min(result) == np.max(result)):
			print 'no file got'
			break
		
		# debug
		# print 'result shape', (result.shape)
		res = result[0, 0, :, :].reshape((h, w)) * 255  #
		
		input_resized = cv2.resize(np.uint8(images_original[i, :, :, :].reshape(h, w) * 255), (480, 360))
		
		# debug
		# print 'min,max:%f %f' % (np.min(images_original[i, :, :, :]), np.max(images_original[i, :, :, :]))
		
		output_resized = cv2.resize(np.uint8(res), (480, 360))
		
		input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_GRAY2RGB)
		
		color = tools.add_colorOverlay(input_resized, output_resized)
		combined = np.hstack((input_rgb, color))
		
		# debug
		# cv2.imwrite('../testbench/frame_res_tmp.jpg', np.uint8(res))
		print '# show segment result for frame', im_i
		demo_result_imgs(res, combined, color, im_i, save_file=False, demo=True)
	
	#############################################################################
	# tmp_time0=time.time() #time for a loop 5.88762402534,5.81136989594 start from here
	# CLASSIFICATION
	
	# TODO evaluate frame_tensor to get detected ROI
	# cv2.imwrite('../testbench/frame_crop_color.jpg',frame_crop_color)
	# frame_tensor = ROI('../testbench/frame_res_tmp.jpg', frame_crop_color, 320, 320, frame_i=video_frame_i,
	#                   save_file=False)
	image_crop_roi, screen_out = ROI(res, im_crop_color, h, w, im_i=im_i, save_file=False)
	# time for ROI 5.74937677383
	# tmp_time0 = time.time() #time for a loop 0.0359718799591 start from here
	# print 'time for write',tmp_time2-tmp_time1 # 0.00110197067261, 7.58029007912(no write), 7.33648395538(use write)
	return image_crop_roi, screen_out


def EVALUATE_VIDEO_seg_and_classify(VIDEO_FILE, num_class):  # (v)
	# load seg model
	set_image_dim_ordering(dim_ordering='th')  # if not configured in
	model = load_model("../testbench/bg/" + bg_model)
	print 'loaded model', bg_model
	
	n_classes = num_class
	video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
	
	video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
	# hy: propID=1 means 0-based index of the frame to be decoded/captured next
	
	if not video.isOpened():
		print "cannot find or open video file"
		exit(-1)
	
	# hy: initialize confmatrix
	confMat2_TEST_Video = np.zeros((2, 2), dtype=np.float)
	
	## Reading the video file frame by frame
	video_frame_i = 0
	while True:  # and video_frame_i < 850:
		
		ret, frame = video.read()
		if ret:  # time for a loop 7.28790903091 start from here
			h = frame.shape[0]
			w = frame.shape[1]
			video_frame_i += 1
		# print '\n\n########################################### start, frame', video_frame_i
		# print 'frame shape h,w:', h, w  # 1536 2304
		else:
			break
		
		if video_frame_i % 2 == 0:
			
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
			
			frame_crop_roi, screen_out = do_segment(model, frame_crop_color, frame_crop, video_frame_i, 320, 320)
			
			# time elapsed for classification 0.0430190563202
			######################### get classified result start#################################
			#do_classification(frame, video_frame_i, frame_crop_roi, frame_crop, confMat2_TEST_Video, crop_y1, crop_y2,
			#                  crop_x1,crop_x2, n_classes, screen_out)
			
			new_graph = tf.Graph()
			with tf.Session(graph=new_graph) as sess2:
				# load classifier model
				# sess, saver = tools.load_classifier_model(sess, '../testbench/6classes/', classifier_model=classifier_model)
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir="../logs/")
				saver = tf.train.import_meta_graph(classifier_model)  # (v)
				
				ckpt.model_checkpoint_path = classifier_model[:-5]
				
				if ckpt and ckpt.model_checkpoint_path:
					saver = tf.train.Saver()
					saver.restore(sess2, ckpt.model_checkpoint_path)
					print "Evaluation with model", ckpt.model_checkpoint_path
				else:
					print 'not found model'
				
				tensorImgIn, test_labels = get_tensor(video_frame_i, frame_crop_roi, n_classes, cvtcolor=True, screen_out=True)
				
				RES = get_classify_result(sess2, tensorImgIn, test_labels, video_frame_i, frame, frame_crop,
			                    crop_y1, crop_y2, crop_x1, crop_x2, screen_out=False)
			
				target = tools.get_ground_truth_label(video_frame_i, default=False)
				
				confMat1_TEST, confMat2_TEST = tools.process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop,
				                                                 SAVE_CorrectClassified, SAVE_Misclassified, video_frame_i,
				                                                 target=target)
			
			######################### Tensorflow
			'''
   # frame_demo = imutils.resize(frame, width = min(1200, settings.w_resize * 30)) #hy: choose a smaller window size
   cv2.imshow("Demo", frame_demo)
   cv2.waitKey(30)
   '''
			######################################
			# cv2.waitKey(6000)
		if cv2.waitKey(1) & 0xFF == ord('q'):  # hy:press key-q to quit
			break
			# tmp_time3=time.time()
			# cv2.waitKey(10)
			# print 'time for a loop',tmp_time3-tmp_time0 #7.11944699287


def EVALUATE_IMAGE_seg_and_classify(h, w, file_i=1):
	# load test data
	read_path = '../Test_Images/testpkg3_unet/'
	ImageType = '.jpg'
	files = [s for s in os.listdir(read_path) if ImageType in s]
	
	# load seg model
	set_image_dim_ordering(dim_ordering='th')  # if not configured in
	model = load_model("../testbench/bg/" + bg_model)
	print 'loaded model', bg_model
	
	# debug
	# model.summary()
	
	for im_ori, im_i in zip(files, xrange(len(files))):
		im_ori = cv2.imread(read_path + im_ori)
		im_crop = im_ori
		image_crop_roi, screen_out = do_segment(model, im_ori, im_crop, im_i, h, w)
		cv2.imshow('seg_img_res', image_crop_roi)
		cv2.waitKey()
	
	# return image_crop_roi,screen_out


def seg_image_2c(h, w, file_i=1):
	print 'start evaluation, model', bg_model
	images, mask = tools.import_data_unet_2c("../tmp/distor_in/input/", "cad_%03d.jpg", "cad_m_%03d.jpg", h, w, 3, file_i,
	                                         do_Flipping=False)
	# params: loadData(data_path, file_img, file_mask, h, w, maxNum, do_Flipping=False)
	test_num = (images.shape)[0]
	print 'number of test images', test_num, ', shape', images.shape  # (3, 1, 320, 320)
	
	images = images[1:-1, :, :, :]  #
	print '2-shape of test images', images.shape  # (1, 1, 320, 320)
	
	set_image_dim_ordering(dim_ordering='th')  # if not configured in
	try:
		model = load_model("../testbench/bg/" + bg_model)
		print 'loaded model', bg_model
	except Exception as e:
		print '[Hint-e]', e
	
	# debug
	# model.summary()
	
	mean, stdev = tools.calc_mean_stdev(images)
	
	images_original = images.copy()
	# print '3-number of test images', images_original.shape
	images = (images - mean) / stdev
	
	# print 'number of test images', (images.shape), ', ... ', images.shape
	
	for i in range(0, images.shape[0]):
		start = time.time()
		result = model.predict(images[i, :, :, :].reshape(1, 1, h, w), batch_size=1)  # old
		end = time.time()
		print(end - start)
		print 'Test image', i + 1, ", Min/Max: %f %f" % (np.min(result), np.max(result))
		print 'result shape', (result.shape)
		res = result[0, 0, :, :].reshape((h, w)) * 255  # old
		
		input_resized = cv2.resize(np.uint8(images_original[i, :, :, :].reshape(h, w) * 255), (480, 360))
		print 'min/max:%f %f' % (np.min(images_original[i, :, :, :]), np.max(images_original[i, :, :, :]))
		
		# debug
		# cv2.imshow("input_resized_window1", input_resized)
		# cv2.waitKey(10000)
		
		output_resized = cv2.resize(np.uint8(res), (480, 360))
		
		input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_GRAY2RGB)
		
		# cv2.imshow("input_rgb_window2", input_rgb)
		# cv2.waitKey(10000)
		
		color = tools.add_colorOverlay(input_resized, output_resized)
		combined = np.hstack((input_rgb, color))
		save_file = True
		if save_file:
			cv2.imwrite('../testbench/res.jpg', np.uint8(res))
			# cv2.imwrite('../testbench/res_' + bg_model[:-5] + '%03d.jpg' % i, np.uint8(res))
			cv2.imwrite("../testbench/combined_%03d.jpg" % i, combined)
			cv2.imwrite("../testbench/color_%03d.jpg" % i, color)
		else:
			cv2.imshow('res', np.uint8(res))
		# cv2.imwrite('combined', combined)
		# cv2.imwrite('color', color)
		##########################################################
		# res2 = result[0,0, :, 1].reshape((320,320))*255
		# back = result[0,0, :, 2].reshape((320, 320)) * 255
		# cv2.imshow("S", np.uint8(res))
		# cv2.imshow("S2", np.uint8(res2))
		# cv2.imshow("Bg", np.uint8(back))
		input = np.uint8(((images[i, :, :, :] * stdev + mean) * 255).reshape(320, 320))
		cv2.imshow("input", input)
		
		cv2.waitKey(10)
		return np.uint8(res)


def generate_training_data(w, h, file_i, save_file=False):
	cv2.waitKey(30)
	obj_area = seg_image_2c(320, 320, file_i=file_i)
	ori_img = '../tmp/distor_in/input/cad_%03d' % file_i + '.jpg'
	ROI('../testbench/res.jpg', ori_img, w, h, frame_i=file_i, save_file=save_file)


if generate_tr_data:
	for train_data_i in xrange(1, 39):
		generate_training_data(320, 320, file_i=train_data_i, save_file=False)


def eva_image_6c(h, w):
	print 'start evaluation'
	images, mask = tools.import_data_unet_6c("../Test_Images/testpkg3_unet/", "cadbg_%03d.jpg", "fgbg_%03d.jpg", h, w, 3,
	                                         do_Flipping=False)
	# params: loadData(data_path, file_img, file_mask, h, w, maxNum, do_Flipping=False)
	test_num = (images.shape)[0]
	print 'number of test images', test_num, ', shape', images.shape  # (3, 1, 320, 320)
	
	set_image_dim_ordering(dim_ordering='th')  # if not configured in
	
	try:
		model = load_model("../testbench/bg/" + bg_model)
		print 'loaded model', bg_model
	except Exception as e:
		print '[Hint-e]', e
	
	# debug
	# model.summary()
	
	######################
	mean = np.mean(images)
	print 'mean', mean
	stdev = np.std(images)
	print 'stdev', stdev
	######################
	
	images_original = images.copy()
	# print '3-number of test images', images_original.shape
	images = (images - mean) / stdev
	
	for i in range(0, images.shape[0]):
		start = time.time()
		ch = 3
		result = model.predict(images[i, :, :, :].reshape(1, ch, h, w), batch_size=1)  #
		end = time.time()
		print(end - start)
		print 'Test image', i + 1, ", Min/Max: %f %f" % (np.min(result), np.max(result))
		print 'result shape', (result.shape)
		res = result[0, 0, :, :].reshape((h, w)) * 255  # old
		
		print 'images_original shape', images_original.shape
		input_resized = cv2.resize(np.uint8(images_original[i, :, :, :].reshape(h, w, 3) * 255), (480, 360))
		# input_resized = cv2.resize(np.uint8(images_original[i, :, :, :].reshape(h, w) * 255), (480, 360)) #old
		# print 'min/max:%f %f' % (np.min(images_original[i, :, :, :]), np.max(images_original[i, :, :, :]))
		
		output_resized = cv2.resize(np.uint8(res), (480, 360))
		
		# input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_GRAY2RGB)
		
		# cv2.imshow("input_rgb_window2", input_rgb)
		# cv2.waitKey(10000)
		
		# color = tools.add_colorOverlay(input_resized, output_resized)
		# combined = np.hstack((input_rgb, color))
		# cv2.imwrite('../testbench/res_' + bg_model[:-5] + '%03d.jpg' % i, np.uint8(res))
		
		# cv2.imwrite("../testbench/combined_%03d.jpg" % i, input_resized)
		# cv2.imwrite("../testbench/combined_%03d.jpg" % i, combined)
		# cv2.imwrite("../testbench/color_%03d.jpg" % i, color)
		
		# debug
		cv2.imshow("input_resized_window1", input_resized)
		cv2.imshow("out_window1", output_resized)
		cv2.imshow("res_window1", np.uint8(res))
		cv2.waitKey(500)
		
		##########################################################
		# res2 = result[0,0, :, 1].reshape((320,320))*255
		# back = result[0,0, :, 2].reshape((320, 320)) * 255
		# cv2.imshow("Standing", np.uint8(res))
		# cv2.imshow("Sitting", np.uint8(res2))
		# cv2.imshow("Background", np.uint8(back))
		# cv2.imshow("input", np.uint8(((images[i, :, :, :] * stdev + mean) * 255).reshape(320, 320)))
		
		cv2.waitKey(30)


##########################################################################################################
if EVALUATE_VIDEO == 1:
	
	test_video_custom = 1
	if test_video_custom == 1:
		print '####################################'
		print 'start evaluation with seg-net and classification'
		print bg_model, classifier_model
		video_list = ['full/']
		TestFace = 'full'
		video_window_scale = 1
		VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
		print VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label
		
		EVALUATE_VIDEO_seg_and_classify(VIDEO_FILE, 6)
	
	else:
		video_list = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
		
		for video_index in xrange(len(video_list)):
			TestFace = video_list[video_index][:-1]  # all # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
			print 'Test face:', TestFace
			
			# TestFace = settings.LABELS[video_index][:-1] #'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
			VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, scale=2)
			EVALUATE_VIDEO_seg(VIDEO_FILE, 6)
			print 'test face:', TestFace, 'done\n'
	
	'''
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

  test_video_custom = 1
  if test_video_custom == 1:
   print '####################################'
   print 'start evaluation with seg-net and classification'
   print bg_model, classifier_model
   video_list = ['full/']
   TestFace = 'full'
   video_window_scale = 1
   VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, video_window_scale)
   print VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label

   EVALUATE_VIDEO_seg(sess, VIDEO_FILE, 6)

  else:
   video_list = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']

   for video_index in xrange(len(video_list)):
    TestFace = video_list[video_index][:-1]  # all # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
    print 'Test face:', TestFace

    # TestFace = settings.LABELS[video_index][:-1] #'vorn'  #'hinten' # full, 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
    VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label = tools.set_video_window(TestFace, scale=2)
    EVALUATE_VIDEO_seg(VIDEO_FILE, 6)
    print 'test face:', TestFace, 'done\n'
 '''

if EVALUATE_IMAGES == 1:
	EVALUATE_IMAGE_seg_and_classify(320, 320, file_i=1)
	# seg_image_2c(320, 320)
	# eva_image_6c(320, 320)
	print("Evaluation done!")

