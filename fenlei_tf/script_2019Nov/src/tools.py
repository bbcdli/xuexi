# hy:change logs are in tensorflow_train.py
#####################################################################################################
#import ImageDraw
#import ImageFilter
#import ImageOps
import time
import random
import glob
from functools import wraps
from random import randint
import os
import sys
import datetime
import settings  # hy: collection of global variables
import prep_image
import cv2
import numpy as np
import tensorflow as tf
from sklearn import datasets
import math
import imutils
from PIL import Image  # hy: create video with images

# activate global var
settings.set_global()
start_time = time.time()
# http://lvdmaaten.github.io/tsne/ visualization
## Train or Evaluation
############################################################
# act_min = 0.80
# act_max = 0.93
# add_data = 0 #initial
# area_step_size_webcam = 20 #479 #200
optimizer_type = 'GD'  # 'adam' #GD-'gradient.descent'
# n_hidden = 2000

# SAVE_Misclassified     = 0
# SAVE_CorrectClassified = 0

# GENERATE_FILELIST = 0
# log_on = False
DEBUG = 0
ImageType = '.jpg'
# Network Parameters
# n_input = settings.h_resize * settings.w_resize  #hy
n_classes = len(settings.LABELS)  # hy: adapt to lego composed of 6 classes. Cifar10 total classes (0-9 digits)
dropout = 0.8  # Dropout, probability to keep units

# Data
LABEL_LIST = settings.data_label_file
LABEL_PATH = settings.data_label_path

LABEL_LIST_TEST = settings.test_label_file
LABEL_PATH_TEST = settings.test_label_path

LABELS = settings.LABELS  # hy
LABEL_names = settings.LABEL_names  # hy

# auto-switches  #########################
# result_for_table = 0

# hy:add timestamp to tensor log files
from datetime import datetime

tensorboard_path = '/tmp/Tensorboard_data/sum107/' + str(datetime.now()) + '/'


class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		from datetime import datetime
		str_log = optimizer_type
		self.log = open(datetime.now().strftime('log_%Y_%m_%d_%H_%M' + str_log + '.log'), "a")
	
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
	
	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


# if log_on and (RETRAIN or CONTINUE_TRAIN or TEST_with_Video):
#  sys.stdout = Logger()


def prepare_list(image_list_file, image_label_path):
	filelist = sorted(glob.glob(image_label_path))  # hy: sorted is used for doing active fields analysis
	Output_file = image_list_file
	
	#print 'image_label_path',image_label_path
	
	if DEBUG:
		print(filelist)
	
	file_name_label = []
	
	# method 1
	# label_list = os.listdir(class_PATH)
	# print "class list:", label_list
	
	# method 2 same result as 1
	# class_label = next(os.walk(class_PATH))[1]
	# print "class_label:", class_label
	if DEBUG:
		if os.path.isfile(Output_file) == False:
			print("file not found, please create one empty file first")
			open(Output_file)
		else:
			print("file found OK")
	
	for filename in filelist:
		class_index = 0
		for label in LABELS:  # hy search files under this path
			# label = class_PATH + label  #hy ../Data/2
			if str.find(filename, label) != -1:  # hy find all lines containing /Data/class_index
				file_name_label.append(filename + " " + str(class_index))
			# print file_name_label
			# else:
			#    print 'no folder found'
			class_index = class_index + 1
	
	lines = "\n".join(file_name_label)
	
	# write lines into the file
	with open(Output_file, 'w') as f:
		f.writelines(lines)
	
	# print "first line:"
	# with open(Output_file, 'r') as f:
	#    plines = [next(f) for x in range(1)]
	#    print plines
	
	if DEBUG:
		# method 1
		print("method 1: file length:", sum(1 for line in open(Output_file)))
		
		# method 2
		with open(Output_file) as f:  # use "with sth as" to define a file
			print("method 2: file length:", sum(1 for line in f))
		
		# method 3
		with open(Output_file) as f:
			file_length = len(f.readlines())
			print("method 3: file length:", file_length)
	
	print('file list is created.', image_list_file, 'path', image_label_path)

def read_images(filelist, random_read=True):
	# read file generated in .txt with prepare.py
	# print 'file list',filelist
	# filelist = '../FeatureMapList.txt'
	
	lable_file = open(filelist, 'r')
	lines = lable_file.readlines()
	
	if random_read == True:
		# make the file order in random way, this makes training more efficiently
		random.shuffle(lines)
		print('loading images in random order')
	else:
		print('loading sorted images')
	
	images = []
	labels = []
	files = []
	label_sum = 0
	
	for item in lines:
		filename, label = item.split()
		
		#print 'filename',filename
		
		im = cv2.imread(filename)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im = transform_to_one_size(im)
		
		'''
		# im = imutils.resize(im, width = 72, height = 72) # w=146, h=121
		im_gray = imutils.resize(im, width=settings.w_resize, height=settings.h_resize)  # w=146, h=121
		# im=imutils.resize(cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY),width=42,height=24)
		# add ###################################################################################
		# resize to fit classifier model
		im_resize_gray = cv2.resize(im_gray, (settings.h_resize, settings.w_resize))
		im = np.float32(im_resize_gray.reshape(settings.h_resize, settings.w_resize))
		
		im = np.asarray(im, np.float32)
		'''
		
		# im = im.reshape((-1, im.size))
		# im = np.expand_dims(np.array(im), 2).astype(np.float32)
		########################################################################################
		
		# im=np.asarray(im, np.float32) #old
		
		images.append(im)
		labels.append(label)
		files.append(filename)
		label_sum = int(label) + label_sum
	
	images_np = np.asarray(images)
	labels_np = np.asarray(labels)
	#  #print "sum of labels:",label_sum #debug #hy need to correct
	#  tmp = Image.fromarray(images_np[2],'L')
	#  tmp.show()
	
	# print "h_resize = ", settings.h_resize
	# print "images_np array shape:",images_np.shape
	
	# print "im shape:", im.shape
	# print "total files:", len(files)
	return (images_np, labels_np, files)

# main functionality testing
# i,l,f = read_images('fileList.txt')
# print i.shape
# tmp = Image.fromarray(i[l])
# tmp = i[46]
# tmp = tmp.astype(npuint8)
# print tmp.shape
# cv2.imshow("test",tmp)
# print l(46)
# tmp.show()
# cv2.waitKey(0)


def load_base_images(feed_train_data=False, feed_test_images=False, CREATE_FILELIST=False, CREATE_TESTLIST=False):
	images = []
	targets = []
	f = []
	
	if CREATE_FILELIST:
		prepare_list(LABEL_LIST, LABEL_PATH)
	
	if CREATE_TESTLIST:
		prepare_list(LABEL_LIST_TEST, LABEL_PATH_TEST)
	
	if feed_train_data:
		images, targets, f = read_images(LABEL_LIST, random_read=True) ## hy: require read in random
	
	if feed_test_images:  ## hy: require read in order
		images, targets, f = read_images(LABEL_LIST_TEST, random_read=False)
	# tools.read_image_output_slices(LABEL_LIST_TEST) #hy: get slices for activation analysis
	return images, targets, f


def read_test_images(filelist):
	# read file generated in .txt with prepare.py
	
	lable_file = open(filelist, 'r')
	lines = lable_file.readlines()
	# make the file order in random way, this makes training more efficiently
	random.shuffle(lines)
	
	images = []
	labels = []
	files = []
	label_sum = 0
	
	for item in lines:
		filename, label = item.split()
		files.append(filename)
		labels.append(label)
	
	labels_np = np.asarray(labels)
	print('read_test_image() done')
	return (labels_np, files)

def read_image_output_slices(filelist):
	lable_file = open(filelist, 'r')
	lines = lable_file.readlines()
	# make the file order in random way, this makes training more efficiently
	random.shuffle(lines)
	
	images = []
	labels = []
	files = []
	label_sum = 0
	
	for item in lines:
		filename, label = item.split()
		
		# print filename
		_im = cv2.imread(filename)
		_im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
		# im = imutils.resize(im, width = 72, height = 72) # w=146, h=121
		(w, h) = _im.shape  # hy:get suitable value for width
		num_of_parts_per_row = 4
		for row in range(0, num_of_parts_per_row):
			for col in range(0, num_of_parts_per_row):
				im = _im[72 / num_of_parts_per_row * col:72 / num_of_parts_per_row * (col + 1),
				     72 / num_of_parts_per_row * row:72 / num_of_parts_per_row * (row + 1)]  # x1:x2,y1:y2
				im = np.asarray(im, np.float32)
				
				images.append(im)
				labels.append(label)
				files.append(filename)
				label_sum = int(label) + label_sum
	
	images_np = np.asarray(images)
	labels_np = np.asarray(labels)
	print('read_test_image() done')
	return (images_np, labels_np, files)


def read_images_online_FIRST(filelist='', random_read=True, Anti_clockwise=0, Clockwise=0, Rotation_Angle=0, Flip_X=0,
                       Flip_Y=0, noise_level=0, step=0):
	print(filelist, random_read, Anti_clockwise, Clockwise, 'angle', Rotation_Angle, 'flip', Flip_X, Flip_Y, noise_level, step)
	save_path = settings.data
	show = False
	lable_file = open(filelist, 'r')
	lines = lable_file.readlines()
	
	if random_read == True:
		# make the file order in random way, this makes training more efficiently
		random.shuffle(lines)
		print('loading images in random order')
	else:
		print('loading sorted images')
	
	images = []
	labels = []
	files = []
	
	for item, index in zip(lines, range(len(lines))):
		filename, label = item.split()
		name = os.path.basename(filename)[:-len(ImageType)]
		im = cv2.imread(filename)
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		
		im_base = transform_to_one_size(im_gray)
		
		images, labels, files = add_to_images_list(im_base, label, filename, images, labels, files)
		cv2.imshow('im_base', im_base)
		
		############ add online data begin
		# Rotation_Angle = randint(15, 170)
		# noise_level = 0.01 * randint(1, 2)
		### Rotation start ##############################################################
		def add_rotation(im, Rotation_Angle, label, filename, images, labels, files, i=0, save_path='', name='', show=False,
		                 save_file=False):
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				# im = cv2.resize(im,(settings.w_resize,settings.h_resize), interpolation=cv2.INTER_LINEAR)
				# print 'received im for rotation h,w', im.shape[0], im.shape[1]
				rotated = imutils.rotate(im, angle=Rotation_Angle)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('ori' + str(i) + '_' + filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('rota' + str(i) + '_' + str(Rotation_Angle) + filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_rota1_' + str(Rotation_Angle) + ImageType
					cv2.imwrite(filename, im)
				
				# im = cv2.resize(im,(settings.w_resize,settings.h_resize), interpolation=cv2.INTER_LINEAR)
				# print 'received im for rotation h,w', im.shape[0], im.shape[1]
				rotated = imutils.rotate(im, angle=Rotation_Angle + 5)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('ori2' + str(i) + '_' + filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('rota2' + str(i) + '_' + str(Rotation_Angle) + filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_rota2_' + str(Rotation_Angle) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
			
			if Clockwise == 1 and Rotation_Angle != 0:
				# Clockwise
				rotated_tmp = cv2.resize(im, (settings.w_resize + 40, settings.h_resize + 20), interpolation=cv2.INTER_LINEAR)
				rotated_tmp = imutils.rotate(rotated_tmp, angle=Rotation_Angle * -1)
				rotated = rotated_tmp[10:settings.h_resize + 10, 20:settings.w_resize + 20]
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				# rotated = imutils.resize(rotated, width=settings.w_resize, height=settings.h_resize)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('c_ori1' + str(i) + '_' + filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('c_rota1' + str(i) + '_' + str(Rotation_Angle) + filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_c_rota1_' + str(Rotation_Angle) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
			
			return images, labels, files
		
		images, labels, files = add_rotation(im_base, Rotation_Angle, label, filename, images, labels, files, index,
		                                     save_path,
		                                     name, show=show, save_file=False)
		
		def add_flip(im, label, filename, images, labels, files, i=0, save_path='', name='', show=False, save_file=False):
			### Flipping begin ##############################################################
			for flipType in range(-1, 2):
				flipped = cv2.flip(im, flipType)
				images, labels, files = add_to_images_list(flipped, label, filename, images, labels, files)
				if show:
					cv2.imshow('flip' + str(i) + str(flipType + 1), flipped)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_flip' + str(flipType + 1) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
			return images, labels, files
		
		images, labels, files = add_flip(im_base, label, filename, images, labels, files, index, save_path, name, show=show,
		                                 save_file=False)
		### add noise begin ##############################################################
		if noise_level != 0:
			img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			
			# img = img.astype(np.float32)
			img_noised = img_gray + np.random.rand(settings.w_resize, settings.h_resize) * noise_level
			img_noised = (img_noised / np.max(img_noised)) * 255
		
		### add noise end    ##############################################################
		
		def do_augment_data(im, label, filename, images, labels, files, h, w, show=False, save_file=False):
			name = os.path.basename(filename)[:-len(ImageType)]
			save_path = settings.data
			# im = cv2.imread(filename)
			im_proportions = []
			ims_resize = []
			
			# first cut different proportion from whole image
			def cutCVimg(im, ori_h, ori_w, ratio):
				w_new = int(ori_w * ratio * 0.78)  # simulate different detection area sizes
				h_new = int(ori_h * ratio)
				im = im[0:h_new, ori_w - w_new:w_new]  # y1:y2,x1:x2
				# im = im[ori_h - h_new:h_new, ori_w - w_new:w_new]  # y1:y2,x1:x2
				# resize them to seg output size hxw (simulate)
				im = cv2.resize(np.uint8(im), (h, w))
				return im
			
			# create patches
			for r in range(7, 9):
				im_proportion = cutCVimg(im, im.shape[0], im.shape[1], r * 0.125)  # [0]h,[1]w for rectangular use 0.12
				im_proportions.append(im_proportion)
				for im in im_proportions:
					if show:
						cv2.imshow('proportion' + str(r), im)
						cv2.waitKey(10)
			
			# second, add to transform_to_one_size
			for im, i in zip(im_proportions, range(len(im_proportions))):
				# im = cv2.resize(np.uint8(im), (h, w))
				# im = imutils.resize(im, width=settings.w_resize)
				im_base = transform_to_one_size(im)
				ims_resize.append(im)
				if show:
					cv2.imshow('resize_' + str(i), im)
					cv2.waitKey()
				if save_file:
					cv2.imwrite(save_path + name + '_' + str(i) + ImageType, im)
				# third, add rotation and add to images list
				# for im_base, i in zip(ims_resize, range(len(ims_resize))):
				images, labels, files = add_rotation(im_base, Rotation_Angle, label, filename, images, labels, files, i,
				                                     save_path, name, show=show, save_file=False)
				
				# next, add all flipping and add to images list
				# for im_base, i in zip(ims_resize, range(len(ims_resize))):
				images, labels, files = add_flip(im_base, label, filename, images, labels, files, index, save_path, name,
				                                 show=show,
				                                 save_file=False)
				'''
				if show:
					cv2.imshow('flip' + str(i) + str(flipType + 1), flipped)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_flip' + str(flipType + 1) + ImageType
					print 'filename', filename
					cv2.imwrite(filename, im)
				'''
			return images, labels, files
		
		if 'full' in filename:  # hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/
			images, labels, files = do_augment_data(im_gray, label, filename, images, labels, files, 320, 320, show=False,
			                                        save_file=False)
		
		'''
		if index > int(0.3*len(lines)) and index < int(0.4 * len(lines)):
			############ add online data begin
			# Rotation_Angle = randint(15, 170)
			# noise_level = 0.01 * randint(1, 2)
			### Rotation start ##############################################################
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				rotated = imutils.rotate(im, angle=Rotation_Angle+3)
				images.append(rotated)
				labels.append(label)
				files.append(filename)

		if index > int(0.5*len(lines)) and index < int(0.6 * len(lines)):
			############ add online data begin
			# Rotation_Angle = randint(15, 170)
			# noise_level = 0.01 * randint(1, 2)
			### Rotation start ##############################################################
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				rotated = imutils.rotate(im, angle=Rotation_Angle+23)
				images.append(rotated)
				labels.append(label)
				files.append(filename)

		'''
	
	images_np = np.asarray(images)
	labels_np = np.asarray(labels)
	# print len(images),len(labels),len(files)
	
	return (images_np, labels_np, files)


###########
def transform_to_one_size(im):
	# resize to fit classifier model
	im = cv2.resize(im, (settings.h_resize, settings.w_resize))
	im = imutils.resize(im, width=settings.w_resize, height=settings.h_resize)
	#im = imutils.resize(im_resize_gray, width=settings.w_resize, height=settings.h_resize)
	#im_resize_gray = cv2.resize(im, (settings.h_resize, settings.w_resize))
	
	return im

def add_to_images_list(im, label, filename, images, labels, files):
	im = np.asarray(im, np.float32)
	im = np.float32(im.reshape(settings.h_resize, settings.w_resize))
	
	images.append(im)
	labels.append(label)
	files.append(filename)
	return images, labels, files

############
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
	#test_image = cv2.resize(pre_tensor, (settings.h_resize, settings.w_resize))
	#test_image = imutils.resize(test_image, width=settings.w_resize, height=settings.h_resize)
	#test_image = np.asarray(test_image, np.float32)
	
	test_image = transform_to_one_size(pre_tensor)
	
	test_image = np.asarray(test_image, np.float32)
	test_image = np.float32(test_image.reshape(settings.h_resize, settings.w_resize))
	
	tensorImgIn = test_image.reshape((-1, test_image.size)) # can be other size
	tensorImgIn = np.expand_dims(np.array(tensorImgIn), 2).astype(np.float32)
	tensorImgIn = tensorImgIn / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
	
	test_labels = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
	return tensorImgIn, test_labels


def do_statistics(confMat1, confMat2, data_length):
	# for i in range(0, len(settings.LABELS)):
	#	confMat1[i, :] = confMat1[i, :] / np.sum(confMat1[i, :])
	print_label_title()
	print(confMat1)
	tp = confMat2[0, 0]
	tn = confMat2[1, 1]
	overall_acc = round(tp / data_length, 2)
	print('TEST overall acc:', overall_acc)
	
	return overall_acc


def process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop, SAVE_CorrectClassified, SAVE_Misclassified, im_i,
                target=1):
	# print '\ntarget,RES', target, RES
	if RES == target:
		label2_TEST = 0
		pred2_TEST = 0
		name_str = settings.CorrectClassified + "/frame_crop%d.jpg" % im_i
		SAVE_CorrectClassified_frame(name_str, frame_crop, SAVE_CorrectClassified)  # (v)
	
	else:
		label2_TEST = 1
		pred2_TEST = 1
		name_str = settings.Misclassified + "/frame_crop%d.jpg" % im_i
		SAVE_Misclassified_frame(name_str, frame_crop, SAVE_Misclassified)
	
	confMat1_TEST[target, RES] = confMat1_TEST[target, RES] + 1.0
	confMat2_TEST[label2_TEST, pred2_TEST] = confMat2_TEST[label2_TEST, pred2_TEST] + 1
	return confMat1_TEST, confMat2_TEST


def read_images_online(filelist='', random_read=True, Anti_clockwise=0, Clockwise=0, Rotation_Angle=0, Flip_X=0,
                           Flip_Y=0, noise_level=0, step=0,augment_crop=False):
	print(filelist, random_read, Anti_clockwise, Clockwise,'angle',Rotation_Angle,'flip',Flip_X, Flip_Y, noise_level, step)
	save_path = settings.data
	show = False
	lable_file = open(filelist, 'r')
	lines = lable_file.readlines()
	
	if random_read == True:
		# make the file order in random way, this makes training more efficiently
		random.shuffle(lines)
		print('loading images in random order')
	else:
		print('loading sorted images')
	
	images = []
	labels = []
	files = []
	
	for item, index in zip(lines, range(len(lines))):
		filename, label = item.split()
		name = os.path.basename(filename)[:-len(ImageType)]
		im = cv2.imread(filename)
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		
		im_base = transform_to_one_size(im_gray)
		
		images, labels, files = add_to_images_list(im_base, label, filename, images, labels, files)
		
		############ add online data begin
		# Rotation_Angle = randint(15, 170)
		# noise_level = 0.01 * randint(1, 2)
		### Rotation start ##############################################################
		def add_rotation(im, Rotation_Angle, label, filename, images, labels, files, i=0, save_path='', name='', show=False,
		                 save_file=False):
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				#im = cv2.resize(im,(settings.w_resize,settings.h_resize), interpolation=cv2.INTER_LINEAR)
				#print 'received im for rotation h,w', im.shape[0], im.shape[1]
				rotated = imutils.rotate(im, angle=Rotation_Angle)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('ori' + str(i) + '_' +filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('rota' + str(i) + '_' + str(Rotation_Angle)+filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_rota1_' + str(Rotation_Angle) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)

			  
				#'''
				#im = cv2.resize(im,(settings.w_resize,settings.h_resize), interpolation=cv2.INTER_LINEAR)
				#print 'received im for rotation h,w', im.shape[0], im.shape[1]
				rotated = imutils.rotate(im, angle=Rotation_Angle+136)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('ori2' + str(i) + '_' +filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('rota2' + str(i) + '_' + str(Rotation_Angle)+filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_rota2_' + str(Rotation_Angle) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
				#'''


			if Clockwise == 1 and Rotation_Angle != 0:
				# Clockwise
				rotated_tmp = cv2.resize(im, (settings.w_resize + 40, settings.h_resize + 20), interpolation=cv2.INTER_LINEAR)
				rotated_tmp = imutils.rotate(rotated_tmp, angle=Rotation_Angle * -1)
				rotated = rotated_tmp[10:settings.h_resize + 10, 20:settings.w_resize + 20]
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				# rotated = imutils.resize(rotated, width=settings.w_resize, height=settings.h_resize)
				images, labels, files = add_to_images_list(rotated, label, filename, images, labels, files)
				if show:
					cv2.imshow('c_ori1' + str(i) + '_' + filename[:-8], rotated)
					cv2.waitKey(10)
					
					cv2.imshow('c_rota1' + str(i) + '_' + str(Rotation_Angle) + filename[:-8], rotated)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_c_rota1_' + str(Rotation_Angle) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
				
			return images, labels, files
		
		images, labels, files = add_rotation(im_base, Rotation_Angle, label, filename, images, labels, files, index, save_path,
		                                     name, show=show, save_file=False)
		
		def add_flip(im, label, filename, images, labels, files, i=0, save_path='', name='', show=False, save_file=False):
			for flipType in range(-1, 2):
				flipped = cv2.flip(im, flipType)
				images, labels, files = add_to_images_list(flipped, label, filename, images, labels, files)
				if show:
					cv2.imshow('flip' + str(i) + str(flipType + 1), flipped)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_flip' + str(flipType + 1) + ImageType
					print('filename', filename)
					cv2.imwrite(filename, im)
			return images, labels, files
		
		images, labels, files = add_flip(im_base, label, filename, images, labels, files, index, save_path, name, show=show,
		                                 save_file=False)
		### add noise begin ##############################################################
		if noise_level != 0:
			img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			
			# img = img.astype(np.float32)
			img_noised = img_gray + np.random.rand(settings.w_resize, settings.h_resize) * noise_level
			img_noised = (img_noised / np.max(img_noised)) * 255
		
		### add noise end    ##############################################################
		
		def do_augment_data(im_ori, label, filename, images, labels, files, h, w, show=False, save_file=False):
			print('augment data level 2')
			name = os.path.basename(filename)[:-len(ImageType)]
			save_path = settings.data
			# im = cv2.imread(filename)
			im_proportions = []
			ims_resize = []
			
			# first cut different proportion from whole image
			def cutCVimg(im, ori_h, ori_w, ratio,im_proportions):
				for i in range(2):
					w_new = int(ori_w * ratio * 0.9)  # simulate different detection area sizes
					h_new = int(ori_h * ratio * 0.9)
					
					x = randint(1, int(ori_w*0.1))
					y = randint(1, int(ori_h*0.1))
					#print 'x,y,w_new,h_new',x,y,w_new,h_new
					
					im = im[y:y+h_new,   x:x+w_new]  # y1:y2,x1:x2
					# resize them to seg output size hxw (simulate)
					im = cv2.resize(np.uint8(im), (320, 320))
					im = np.float32(im.reshape(320, 320))
					im_proportions.append(im)
				return im_proportions
			
			# create patches
			#im_ori = im
			for r in range(7, 9):
				im_proportions = cutCVimg(im_ori, im_ori.shape[0], im_ori.shape[1], r * 0.12,im_proportions)  # [0]h,[1]w for rectangular use 0.12
				if show:
					for im in im_proportions:
						cv2.imshow('proportion'+str(r), im)
						cv2.waitKey(10)
			
			# second, add to transform_to_one_size
			for im, i in zip(im_proportions, range(len(im_proportions))):
				#im = cv2.resize(np.uint8(im), (h, w))
				#im = imutils.resize(im, width=settings.w_resize)
				im_base = transform_to_one_size(im)
				
				images, labels, files = add_to_images_list(im_base, label, filename, images, labels, files)
				
				if show:
					cv2.imshow('resize_' + str(i), im)
					cv2.waitKey()
				if save_file:
					cv2.imwrite(save_path + name + '_' + str(i) + ImageType, im)
			
			# third, add rotation and add to images list
				images, labels, files = add_rotation(im_base, Rotation_Angle, label, filename, images, labels, files, i,
				                                     save_path, name, show=show, save_file=False)
			
			# next, add all flipping and add to images list
				images, labels, files = add_flip(im_base, label, filename, images, labels, files, i, save_path, name, show=show,
		                                save_file=False)
				'''
				if show:
					cv2.imshow('flip' + str(i) + str(flipType + 1), flipped)
					cv2.waitKey()
				if save_file:
					filename = save_path + name + '_' + str(i) + '_flip' + str(flipType + 1) + ImageType
					print 'filename', filename
					cv2.imwrite(filename, im)
				'''
		
			return images, labels, files
		
		if 'full' in filename and augment_crop:  # hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/
			# use im_gray as input
			images, labels, files = do_augment_data(im_gray, label, filename, images, labels, files,320, 320, show=False,save_file=False)
		
		
		'''
		if index > int(0.3*len(lines)) and index < int(0.4 * len(lines)):
			############ add online data begin
			# Rotation_Angle = randint(15, 170)
			# noise_level = 0.01 * randint(1, 2)
			### Rotation start ##############################################################
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				rotated = imutils.rotate(im, angle=Rotation_Angle+3)
				images.append(rotated)
				labels.append(label)
				files.append(filename)

		if index > int(0.5*len(lines)) and index < int(0.6 * len(lines)):
			############ add online data begin
			# Rotation_Angle = randint(15, 170)
			# noise_level = 0.01 * randint(1, 2)
			### Rotation start ##############################################################
			if Anti_clockwise == 1 and Rotation_Angle != 0:
				rotated = imutils.rotate(im, angle=Rotation_Angle+23)
				images.append(rotated)
				labels.append(label)
				files.append(filename)

		'''
		
	images_np = np.asarray(images)
	labels_np = np.asarray(labels)
	
	return (images_np, labels_np, files)

#prepare_list(LABEL_LIST, LABEL_PATH)
#read_images_online(LABEL_LIST, random_read
#=True, Anti_clockwise=1, Clockwise=0, Rotation_Angle=5, Flip_X=1, Flip_Y=1, noise_level=0)




# import data
def import_data(add_online=False,augment_crop=False):
	elapsed_time = time.time() - start_time
	print('Elapsed time:', "{:.2f}".format(elapsed_time), 's')
	
	# print "Start training, loading images"
	digits = datasets.load_digits(n_class=n_classes)
	
	if add_online:
		carimages, cartargets, f = read_images_online(LABEL_LIST, random_read
		=True, Anti_clockwise=1, Clockwise=0, Rotation_Angle=5, Flip_X=0, Flip_Y=0, noise_level=0,augment_crop=augment_crop)
	
	else:
		carimages, cartargets, f = read_images(LABEL_LIST, random_read=True)
	
	total_images = len(carimages)
	
	# tmp=Image.fromarray(carimages[0],'L')
	# tmp.show()
	
	carimages = carimages / 255 - 0.5
	
	if DEBUG == 1:
		# You havent yet changed the rest, becareful about sizes and etc.
		print(' Shape of Labels', cartargets.shape)  # hy check if their length is same
	
	digits.images = carimages.reshape((len(carimages), -1))
	
	if DEBUG == 1:
		print("Shape of data set after reshape: ", digits.images.shape)  # hy
	
	digits.images = np.expand_dims(np.array(digits.images), 2).astype(np.float32)
	
	if DEBUG == 1:
		print("Shape of data set after expansion: ", digits.images.shape)  # hy
	
	digits.target = np.array(cartargets).astype(np.int32)
	digits.target = dense_to_one_hot(digits.target, n_classes)
	
	
	if DEBUG == 1:
		print('\n')
		print("5.print target")
		# print digits.target
		
		elapsed_time = time.time() - start_time
		print('Total elapsed time2:', "{:.2f}".format(elapsed_time), 's')
	
	return [total_images, digits, carimages, cartargets, f]
	#return [total_images, digits, carimages, cartargets, f, val2_digits, val2_images, val2_targets, val2_f]


# test
# images_np, labels_np, files = read_images_online(filelist='FileList_TEST1_sing.txt',random_read=True,Anti_clockwise=0,Clockwise=0,Rotation_Angle=0,Flip_X=1,Flip_Y=0,noise_level=0,step=0)
# print 'len of image', len(images_np)

############################################################################################################
# hy: initialize crop frame (interest of area in demo window)
# At the moment, this window has to be adjusted to focus our object.
# Different area shown in focus region leads to different test  result.
############################################################################################################
def set_video_window(TestFace, scale=1):
	VIDEO_FILE = ''
	crop_x1 = 0
	crop_y1 = 0
	area_step_size = 0
	video_label = 1
	if scale == 1:
		if TestFace == 'full':
			# Video file for Demo
			#VIDEO_FILE = '../Test_Videos/dark.avi'
			VIDEO_FILE = '../Test_Videos/LegoTestVideo1.mp4'
			# These settings are for LegoTestVideo1.avi
			crop_x1 = 550  # 550
			crop_y1 = 780  # 300
			area_step_size = 340  # 640
			video_label = 3
		if TestFace == 'hinten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoHinten.avi'
			# These settings are for LegoTestVideoHinten.avi
			crop_x1 = 680
			crop_y1 = 850
			area_step_size = 210
			video_label = 0
		
		if TestFace == 'links':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoLinks.avi'
			crop_x1 = 560
			crop_y1 = 660
			area_step_size = 400
			video_label = 1
		
		if TestFace == 'oben':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoOben.avi'
			crop_x1 = 660
			crop_y1 = 460
			area_step_size = 400
			video_label = 2
		
		if TestFace == 'rechts':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoRechts.avi'
			# These settings are for LegoTestVideoRechts.avi.
			crop_x1 = 550
			crop_y1 = 600
			area_step_size = 450
			video_label = 3
		
		if TestFace == 'unten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoUnten.avi'
			# These settings are for LegoTestVideoUnten.avi
			crop_x1 = 860
			crop_y1 = 600
			area_step_size = 250
			video_label = 4
		
		if TestFace == 'vorn':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoVorn.avi'
			# Following are settings for the video LegoTestVideoVorn.avi
			crop_x1 = 680
			crop_y1 = 790
			area_step_size = 390  # hy: if we do not need the test window to move around in Demo, then it is just the width
			video_label = 5
	
	if scale == 2:
		# version middle size 2
		if TestFace == 'hinten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoHinten.avi'
			# These settings are for LegoTestVideoHinten.avi,
			crop_x1 = 580
			crop_y1 = 750
			area_step_size = 510
			video_label = 0
		
		if TestFace == 'links':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoLinks.avi'
			# These settings are for LegoTestVideoLinks.avi,
			crop_x1 = 360
			crop_y1 = 460
			area_step_size = 880
			video_label = 1
		
		if TestFace == 'oben':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoOben.avi'
			crop_x1 = 660
			crop_y1 = 460
			area_step_size = 500
			video_label = 2
		
		if TestFace == 'rechts':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoRechts.avi'
			# These settings are for LegoTestVideoRechts.avi,
			crop_x1 = 550
			crop_y1 = 600
			area_step_size = 620
			video_label = 3
		
		if TestFace == 'unten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoUnten.avi'
			crop_x1 = 860
			crop_y1 = 600
			area_step_size = 470
			video_label = 4
		
		if TestFace == 'vorn':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoVorn.avi'
			# Following are settings for the video LegoTestVideoVorn.avi
			crop_x1 = 520
			crop_y1 = 590
			area_step_size = 750  # hy: if we do not need the test window to move around in Demo, then it is just the width
			video_label = 5
	
	if scale == 3:
		# version middle size 3
		if TestFace == 'hinten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoHinten.avi'
			# These settings are for LegoTestVideoHinten.avi,
			crop_x1 = 550
			crop_y1 = 780
			area_step_size = 410
			video_label = 0
		
		if TestFace == 'links':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoLinks.avi'
			# These settings are for LegoTestVideoLinks.avi,
			crop_x1 = 260
			crop_y1 = 560
			area_step_size = 480
			video_label = 1
		
		if TestFace == 'oben':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoOben.avi'
			crop_x1 = 700
			crop_y1 = 360
			area_step_size = 600
			video_label = 2
		
		if TestFace == 'rechts':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoRechts.avi'
			# These settings are for LegoTestVideoRechts.avi,
			crop_x1 = 650
			crop_y1 = 800
			area_step_size = 320
			video_label = 3
		
		if TestFace == 'unten':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoUnten.avi'
			crop_x1 = 860
			crop_y1 = 600
			area_step_size = 570
			video_label = 4
		
		if TestFace == 'vorn':
			VIDEO_FILE = '../Test_Videos/LegoTestVideoVorn.avi'
			# Following are settings for the video LegoTestVideoVorn.avi
			crop_x1 = 260
			crop_y1 = 450
			area_step_size = 420  # hy: if we do not need the test window to move around in Demo, then it is just the width
			video_label = 5
	
	# print 'test face and scale:', TestFace, ', ', scale
	return [VIDEO_FILE, crop_x1, crop_y1, area_step_size, video_label]


def get_ground_truth_label(index,default=False):
	if default:
		target = 0
	elif index < 165:
		target = 1
	elif index > 165 and index < 300:
		target = 5
	elif index > 300 and index < 654:
		target = 3
	elif index > 654 and index < 850:
		target = 0
	elif index > 850 and index < 1100:
		target = 1
	elif index > 1100 and index < 1289:
		target = 5
	return target


def print_label_title():
	print(settings.LABEL_names)

def print_label_title_conf():
	print(settings.LABEL_names)

def convert_result(RES):  # 0 hinten, 1 links, 2 oben, 3 rechts, 4 unten, 5 vorn,
	if RES == 0 or RES == 7:
		label_str = 'vorn'
		label_num = 5
	
	if RES == 1 or RES == 3 or RES == 8 or RES == 10:  # 8:rechts/links  12:rechts/links
		label_str = 'rechts'
		label_num = 3
	
	if RES == 2 or RES == 11:
		label_str = 'oben'
		label_num = 2
	
	if RES == 4 or RES == 5 or RES == 6:
		label_str = 'links'
		label_num = 1
	
	if RES == 11:
		label_str = 'unten'
		label_num = 4
	
	if RES == 9 or RES == 10:
		label_str = 'hinten'
		label_num = 0
	return label_str, label_num

def SAVE_Images(filename, filepath):
	OUTPUT_PATH = filepath
	cmd = 'cp ' + filename + ' ' + OUTPUT_PATH
	os.system(cmd)

def SAVE_CorrectClassified_Img(img, save=False):
	if save:
		imgNum = len([name for name in os.listdir(settings.CorrectClassified) if
		              os.path.isfile(os.path.join(settings.CorrectClassified, name))])
		img_name = img
		if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
			SAVE_Images(img_name, settings.CorrectClassified)
		else:
			print('CorrectClassified Folder is full !!!')

def SAVE_CorrectClassified_frame(name_str, img, save=False):
	if save:
		imgNum = len([name for name in os.listdir(settings.CorrectClassified) if
		              os.path.isfile(os.path.join(settings.CorrectClassified, name))])
		img_name = name_str
		# print 'num of files', misNum
		if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
			cv2.imwrite(img_name, img)
		else:
			print('CorrectClassified Folder is full !!!')

def SAVE_Misclassified_Img(img, save=False):
	if save == 1:
		imgNum = len([name for name in os.listdir(settings.Misclassified) if
		              os.path.isfile(os.path.join(settings.Misclassified, name))])
		img_name = img
		if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
			SAVE_Images(img_name, settings.Misclassified)
		else:
			print('Misclassified Folder is full !!!')

def SAVE_Misclassified_frame(name_str, img, save=False):
	if save == 1:
		imgNum = len([name for name in os.listdir(settings.Misclassified) if
		              os.path.isfile(os.path.join(settings.Misclassified, name))])
		img_name = name_str
		# print 'num of files', misNum
		if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
			cv2.imwrite(img_name, img)
		else:
			print('Misclassified Folder is full !!!')

def track_roi(VIDEO_FILE):
	video = cv2.VideoCapture(VIDEO_FILE)  # hy: changed from cv2.VideoCapture()
	# cv2.waitKey(10)
	
	video.set(1, 2)  # hy: changed from 1,2000 which was for wheelchair test video,
	# hy: propID=1 means 0-based index of the frame to be decoded/captured next
	
	if not video.isOpened():
		print("cannot find or open video file")
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
			print('no frame received')
			break
	
	return [track_window]

# hy: input- CNN variable conv and file list; output- learned feature map
def get_feature_map(conv_feature, f, layer):
	save_file = 0
	conv_feature_2D_batch = []
	# batch_xs, batch_ys = digits.images[1:total_images - 1], digits.target[1:total_images - 1]
	## Save Tensor to Files  ##########################################################
	# conv1_feature = sess.run(conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
	np.set_printoptions(threshold=np.nan)  # hy
	num_of_ori_imgs, th, tw, num_of_slices = conv_feature.shape
	print('num of image,h,w,num of slices', num_of_ori_imgs, 'x', th, 'x', tw, 'x', num_of_slices)
	
	# num_of_ori_imgs = len(conv_feature) #hy:debug
	# print 'length feat', num_of_ori_imgs #hy:debug
	
	for im_index in range(num_of_ori_imgs):
		fname = f[im_index]
		fname = os.path.splitext(fname)[0]
		from os.path import basename
		base_name = basename(fname)
		# print fname, '--', base_name #hy for debug
		print('image path name', f[im_index])
		for slice_dim in range(num_of_slices):
			for slice_block in range(num_of_ori_imgs):
				# outfile.write('# Block ' + str(slice_block) + '\n') #hy: debug
				# print '\nnew block ', str(slice_block), 'of hxw: 4x4 *** image ', str(im_index), f[im_index] #hy: debug
				
				for slice_th in range(th):
					for slice_tw in range(tw):
						# hy:convert feature to 2D
						conv_feature_2D = conv_feature[slice_block, :, :, slice_dim]
						
						if save_file == 1:
							# hy:create txt files
							with file('fm_h' + str(settings.h_resize) + 'w' + str(settings.w_resize) + '_' + str(
											im_index) + '-' + base_name + '-' + layer + '-' + str(slice_dim) + '.png',
							          'w') as outfile:  # hy: one txt contains one slice 42x42
								np.savetxt(outfile, conv_feature_2D, fmt='%-2.6f')
								outfile.write('\n')
							
							# hy: debug #
							# outfile.write('# Array shape: {0}\n'.format(conv_feature.shape))
							# outfile.write(f[im_index - 1] + '\n')
							# outfile.write('row ' + str(slice_th) + ' col ' + str(slice_tw) + ' in block ' + str(
							# slice_block) + ' image ' + str(im_index) + '\n')
							# print '\nrow '+ str(slice_th) + ' col '+ str(slice_tw) + ' in block '+ str(slice_block)+' image ' + str(im_index)
			conv_feature_2D_batch.append(conv_feature_2D)
	
	print('feature map saved')  # hy: save feature maps
	return conv_feature_2D_batch

# classifier_model: model path + name
def load_classifier_model(sess, f_path, classifier_model):
	# hy: load saved model with values
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir=f_path)  # "../backupModel/"
	saver = tf.train.import_meta_graph(classifier_model)  # (eva)
	#
	ckpt.model_checkpoint_path = classifier_model[:-5]
	if ckpt and ckpt.model_checkpoint_path:
		print("Evaluation with images, model", ckpt.model_checkpoint_path)
		saver = tf.train.Saver()
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print('not found model')
		print('I-Test with Images starting ...')  # print 'Test with images starting ...', ckpt.model_checkpoint_path
	
	return sess, saver


def EVALUATE_IMAGES_VAGUE(n_classes, img_list):
	LABEL_LIST_TEST = img_list
	# Testing
	cartargets, f = read_test_images(LABEL_LIST_TEST)
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
	
	patch_size = 42  # 227
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
		
		print('processing main test image', f[i])
		
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
		print('slices with path received', slices_rec)
		slices_len = len(slices_rec)
		
		out_sum = np.zeros((1, n_classes), dtype=np.float)
		out_box = np.zeros((1, n_classes), dtype=np.float)
		
		# batch_xs, batch_ys = im, cartargets
		
		# output_im = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
		
		for j in range(0, slices_len, 1):
			print('\nprocessing slice', j, slices_rec[j])
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
			print('slice', j, 'result', sorted_vec)
			print('Image slice', slices_rec[j])
			outbox = outputsub
			out_sum = out_sum + outputsub[0]
		
		# print '\ntp, tn, total number of test images:', tp, ', ', tn, ', ', TEST_length
		# print confMat2_TEST
		print('\nTEST general count:')
		
		print(out_sum)
		print(out_sum / slices_len)
		outbox[0] = out_sum / slices_len
		
		output_im, prob_all = rank_index(outbox[0], test_image_label)
		print('target', test_image_label)
		print('output final prediction', output_im[-1])
		
		RES = int(output_im[-1])
		print('test_image_label', test_image_label)
		
		label = test_image_label
		predict = int(RES)
		
		confMat_m1_TEST[label, predict] = confMat_m1_TEST[label, predict] + 1
		
		count_labels_m[:, test_image_label] = count_labels_m[:, test_image_label] + 1
		
		if int(RES) == int(test_image_label):
			label2_TEST = 0
			pred2_TEST = 0
			
			confMat_m3[:, int(RES)] = confMat_m3[:, int(RES)] + 1
			SAVE_CorrectClassified_Img(f[i], SAVE_CorrectClassified)
		
		
		else:
			label2_TEST = 1
			pred2_TEST = 1
			SAVE_Misclassified_Img(f[i], SAVE_Misclassified)
		
		# print 'Count classified'
		# print_label_title()
		# print confMat1_TEST
		
		confMat_m2_TEST[label2_TEST, pred2_TEST] = confMat_m2_TEST[label2_TEST, pred2_TEST] + 1
		tp = confMat_m2_TEST[0, 0]
		tn = confMat_m2_TEST[1, 1]
		
		print('Count classified m1 - confusion matrix')
		print_label_title()
		print(confMat_m1_TEST)
		
		print('\nCount correctly classified -m3')
		print_label_title()
		print(confMat_m3)
		
		print('tp,np -m2')
		print(confMat_m2_TEST)
		print('Total labels')
		print(count_labels_m)
		
		print('Proportion of correctly classified for detailed analysis')  # ok
		if count_labels_m[:, pos] > 0:
			for pos in range(0, n_classes, 1):
				class_probability_m[:, pos] = confMat_m3[:, pos] / count_labels_m[:, pos]
			print(class_probability_m)
		
		print('TEST overall acc:', "{:.3f}".format(tp / TEST_length))


def rank_index(vector, target, result_for_table):
	tmpLen = len(vector)
	sorted_vec = sorted(range(tmpLen), key=vector.__getitem__)
	prob_all = []
	for index in range(0, tmpLen):
		predict = sorted_vec[tmpLen - 1 - index]
		if predict == target:
			tmpStr = '<-Target\t, Probability:'
			prob_all.append(vector[predict])
		else:
			tmpStr = '\t\t\t, Probability:'
			prob_all.append(vector[predict])
		if result_for_table == 0:
			print(index + 1, ') :', LABELS[predict], tmpStr, vector[predict])
	return sorted_vec, prob_all

def confusion_matrix(labels_onehot, scores, normalized=True):
	n_samples, n_class = scores.shape
	print('n_samples for validation:', n_samples)
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
		print('one_hot_vector:', labels_one_hot[0])
	return labels_one_hot

# Implementing softmax function on the DL output scores, adopted only for 2 classes
# hy: for final output layer using softmax classification
def convert_to_confidence(scores):
	h, w = scores.shape
	output = np.zeros((h, w), dtype=np.float32)
	sum = np.zeros((h, 1), dtype=np.float32)
	# if sum != 0:
	for i in range(0, w):
		sum[:, 0] += np.exp(scores[:, i])
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

import re
def REMOVE_online_Data(step):
	# print 'removing data'
	for i in range(len(LABELS)):
		# cmd = 'rm -r ' + data + folder_list[i] + '/*rotated*.*'  # hy: remove recursively
		# os.system(cmd)
		dir = settings.data + settings.LABELS[i]
		# dir = dir[:-1]
		for f in os.listdir(dir):
			if re.search('/*_st*.*', f) is not None:
				os.remove(os.path.join(dir, f))

def calc_mean_stdev(images):
	mean = np.mean(images)
	print('mean', mean)
	stdev = np.std(images)
	print('stdev', stdev)
	return mean, stdev


############  for u-net     ######################

def import_data_unet_2c(data_path, file_img, file_mask, h, w, maxNum, file_num=1, do_Flipping=False):
	depth = 1
	ch = 1
	print('load data', data_path, file_img, file_mask, h, w, maxNum, do_Flipping)
	images = np.zeros((maxNum * 4, ch, h, w))
	masks = np.zeros((maxNum * 4, ch, h, w))
	
	data_counter = 0
	for i in range(1, maxNum + 1):
		# file_num = 5
		fmask = data_path + file_mask % file_num  #
		# print i, 'of ',maxNum,'join path and current mask file name:',fmask
		
		fimg = data_path + file_img % file_num
		print('\n', i, 'of ', maxNum, 'join path and current img file name:', fimg)
		
		mask = cv2.imread(fmask, 0)
		
		print('read image')
		img = cv2.imread(fimg, 0)
		
		if mask is None or img is None:
			continue
		
		mask = cv2.resize(mask, (h, w))
		img = cv2.resize(img, (h, w))
		
		data_counter += 1
		# debug
		# cv2.imshow("img_window",img)
		# cv2.waitKey(100)
		
		mask = mask.reshape(h, w)
		img = np.float32(img.reshape(h, w))
		# debug
		# print '1-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img), np.std(img))
		
		mask = mask / 255.0
		img = img / 255.0
		# debug
		# print '2-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img),np.std(img))
		
		
		if do_Flipping:
			for fl in range(-1, 2):
				flipped_img = cv2.flip(img, fl)
				flipped_mask = cv2.flip(mask, fl)
				
				images[data_counter, :, :, :] = flipped_img
				masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
				data_counter += 1
		
		images[i, :, :, :] = img
		masks[i, :, :, :] = np.float32(mask > 0)
		
		if i % 100 == 0:
			print('i=', i)
	print('total', data_counter, 'images and', data_counter, 'masks are loaded')
	# return images, masks
	return images[0:data_counter, :, :, :], masks[0:data_counter, :, :, :]


def import_data_unet_6c(data_path, file_img, file_mask, h, w, maxNum, do_Flipping=False):
	d = 3  # d >0: 3-channel, =0: 1-channel, <0:no change
	ch = 3  # 1
	print('load data', data_path, file_img, file_mask, h, w, maxNum, do_Flipping)
	images = np.zeros((maxNum * 4, h, w, ch))
	masks = np.zeros((maxNum * 4, h, w, ch))
	
	data_counter = 0
	for i in range(1, maxNum + 1):
		fmask = data_path + file_mask % i  #
		# print i, 'of ',maxNum,'join path and current mask file name:',fmask
		
		fimg = data_path + file_img % i
		print('\n', i, 'of ', maxNum, 'join path and img file name:', fimg)
		
		mask = cv2.imread(fmask, d)  # d >0: 3-channel, =0: 1-channel, <0:no change
		
		img = cv2.imread(fimg, d)
		
		if mask is None or img is None:
			continue
		
		mask = cv2.resize(mask, (h, w))
		img = cv2.resize(img, (h, w))
		
		data_counter += 1
		# debug
		# cv2.imshow("img_window",img)
		# cv2.waitKey(100)
		
		mask = mask.reshape(h, w, d)
		img = np.float32(img.reshape(h, w, d))
		# debug
		# print '1-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img), np.std(img))
		
		mask = mask / 255.0
		img = img / 255.0
		# debug
		# print '2-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img),np.std(img))
		
		
		if do_Flipping:
			for fl in range(-1, 2):
				flipped_img = cv2.flip(img, fl)
				flipped_mask = cv2.flip(mask, fl)
				
				images[data_counter, :, :, :] = flipped_img
				masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
				data_counter += 1
		
		images[i, :, :, :] = img
		masks[i, :, :, :] = np.float32(mask > 0)
		
		if i % 100 == 0:
			print('i=', i)
	print('total', data_counter, 'images and', data_counter, 'masks are loaded')
	# return images, masks
	return images[0:data_counter, :, :, :], masks[0:data_counter, :, :, :]


def import_data_from_list_unet(path1, h, w, path2='', do_Flipping=False):
	lines1 = sorted(os.listdir(path1))
	lines2 = sorted(os.listdir(path2))
	
	size1 = len(lines1)
	
	size2 = len(lines2)
	
	if path2 is not None:
		print('path2', path2, 'size', size2)
		assert size1 == size2
	ch = 1
	images = np.zeros((4 * size1, ch, h, w))
	masks = np.zeros((4 * size2, ch, h, w))
	data_counter = 0
	
	for i, img, mask in zip(range(size1), lines1, lines2):
		print(img)
		img = cv2.imread(path1 + img, 0)
		mask = cv2.imread(path2 + mask, 0)
		img = cv2.resize(img, (h, w))
		mask = cv2.resize(mask, (h, w))
		print('img shape', img.shape)
		data_counter += 1
		# debug
		# cv2.imshow("img_window",img)
		# cv2.waitKey(100)
		
		mask = mask.reshape(h, w)
		img = np.float32(img.reshape(h, w))
		# debug
		# print '1-min/max:%f %f, mean: %f, std: %f of loaded image' % (np.min(img),np.max(img), np.mean(img), np.std(img))
		
		mask = mask / 255.0
		img = img / 255.0
		
		if do_Flipping:
			for flip_type in range(-1, 2):  # -1:xy, 0:x, 1:y
				flipped_img = cv2.flip(img, flip_type)
				flipped_mask = cv2.flip(mask, flip_type)
				
				images[data_counter, :, :, :] = flipped_img
				masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
				
				data_counter += 1
		
		mask = np.float32(mask > 0)
		
		images[data_counter, :, :, :] = img
		masks[data_counter, :, :, :] = np.float32(mask > 0)
	
	images = images[1:size1 + 1, :, :, :]
	masks = masks[1:size1 + 1, :, :, :]
	
	print('total', len(images), 'images and', len(masks), 'masks are read.')
	return images, masks


# return images[1:data_counter,:,:,:], masks[1:data_counter,:,:,:]

def add_colorOverlay(img_grayscale, mask):
	colorOverlay = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB)
	colorOverlay[:, :, 2] = mask
	return colorOverlay


############  for u-net end ######################

def get_precision(session, im):
	sess = session
	im = np.asarray(im, np.float32)
	
	CONF = 0.20
	
	test_image = im
	
	test_lables = np.zeros((1, n_classes))  # Making a dummy label tp avoid errors as initial predict
	
	# Doing something very stupid here, fix it!
	test_image = im.reshape((-1, im.size))
	
	# print test_image
	# print sess.run(test_image)
	
	test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)
	test_image = test_image / 255 - 0.5  # TODO here is tricky, double check wit respect to the formats
	
	# hy: evaluate
	batch_xs, batch_ys = test_image, test_lables
	
	# output = sess.run("Accuracy:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
	output = sess.run("pred:0", feed_dict={"x:0": batch_xs, "y:0": batch_ys, "keep_prob:0": 1.})
	
	# print("Output for external=",output)
	output = convert_to_confidence(output)  #
	
	return output

