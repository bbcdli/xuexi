# change logs are located in tensor_train.py
from __future__ import division  # for division with float result, must put at the top of this file to make it work
from PIL import Image  # for resize func
import random
from random import randint
import glob
import re
import string
import cv2
import pickle
import os
import imutils
import numpy as np

# import matplotlib.pyplot as plt
import settings

settings.set_global()

SWITCH_data_augmentation = 0
SWITCH_organize_data = 0

do_rotateflipImg = 0
do_augment_data = 0
do_resizeImg = 0  # hy set to 1 to resize image using the function resize defined below
do_crop_Image = 0  # in-distor_in out-inputpatches
crop_num = 1
patch_size = 64  # 42 for cropping, resize, perspective transform
patch_size_h = 64
prepare_active_fields = 0
add_background = 0
do_remove_small_obj = 0

do_convertPCA = 0
do_cropImage_none_overlap = 0
# do_perspectiveTransform   = 0

######### org data
REMOVE_Inputpatches = 0
REMOVE_Inputpatches_outside = 0
REMOVE_TmpData = 0  # both tmp and tmp2
REMOVE_resized = 0
REMOVE_Data = 0
REMOVE_Test_Images = 0

split_Data = 0
split_Inputpatches = 0
split_Test_Images = 0
move_from_tmp_to_patches = 0
move_from_tmp2_to_patches = 0
move_from_tmp2_to_testImg = 0  # sub-folders
copy_from_data_to_patches = 0  # sub-folders
copy_from_patches_to_testImg = 0

copy_from_misclassifed_to_data = 0

############################################
ImageType = '.jpg'
length = len(settings.LABELS)
fileprefix = settings.LABEL_names

# control
if SWITCH_data_augmentation == 0:
	do_resizeImg = 0
	do_augment_data = 0
	# do_crop_Image = 0
	do_remove_small_obj = 0
	do_rotateflipImg = 0
	prepare_active_fields = 0
	add_background = 0
	do_perspectiveTransform = 0
	do_convertPCA = 0
# do_cropImage_none_overlap = 0

if SWITCH_organize_data == 0:
	REMOVE_Inputpatches = 0
	REMOVE_Inputpatches_outside = 0
	REMOVE_TmpData = 0
	REMOVE_Data = 0
	REMOVE_Test_Images = 0
	
	split_Data = 0
	split_Inputpatches = 0
	split_Test_Images = 0
	move_from_tmp_to_patches = 0
	move_from_tmp2_to_patches = 0
	move_from_tmp2_to_testImg = 0
	copy_from_data_to_patches = 0
	copy_from_patches_to_testImg = 0
	copy_from_misclassifed_to_data = 0


###################################################

def do_REMOVE_TmpData():
	print 'removing tmp data'
	for i in xrange(length):
		dir = settings.tmp + settings.LABELS[i]
		for f in os.listdir(dir):
			if re.search('/*.jpg', f) is not None:
				os.remove(os.path.join(dir, f))
	
	dir2 = settings.tmp2
	for f2 in os.listdir(dir2):
		if re.search('/*.jpg', f2) is not None:
			os.remove(os.path.join(dir2, f2))


if REMOVE_TmpData == 1:
	do_REMOVE_TmpData()

#########################################################
# Distortion
# resize, Transition, Rotation 5 Degree, flipping, noise, histogram Eq
########################################################
# hy: config for rotateflipImg()
# OUTPUT_tmp = '../tmp/'
# hy: define labels locally, if only the classes selected need to be augmented
# LABELS = ['hinten', 'links', 'oben', 'rechts', 'unten', 'vorn','neg']
Clockwise = 0
Rotation_Angle = 141  # 90,180,270
Anti_clockwise = 1
Flip_X = 1  # 1 #hy: rechts can have flip_x,

Flip_Y = 1
Flip_XY = 0
do_perspectiveTransform_in = 0
noise_level = 0
step = 1
Aspect_Factor = 0.2
random_add = 4

tmp_PATH = '../tmp/tmp2/'  # copy of original training data (before distortions)
DEBUG = False  #
#
# INPUT_PATCHES_PATH = '../tmp/input_patches/'
INPUT_PATCHES_PATH = '../tmp/tmp2/'
# INPUT_PATCHES_PATH = settings.data
DATA_PATH = settings.data  # save to path


def rotateflipImg(Rotation_Angle=Rotation_Angle, Flip_X=Flip_X, noise_level=noise_level, step=step):  # do_rotateflipImg
	# def rotateflipImg(Rotation_Angle,Flip_X,noise_level,step):
	
	print 'doing rotation or flipping', Rotation_Angle, ',', Flip_X, ',st', step, \
		',noise=', noise_level, ',do_perspectiveTransform_in=', do_perspectiveTransform_in
	
	print 'removing tmp data'
	for i in xrange(len(settings.LABELS)):
		dir = settings.tmp + settings.LABELS[i]
		for f in os.listdir(dir):
			if re.search('/*.jpg', f) is not None:
				os.remove(os.path.join(dir, f))
	
	filelist = glob.glob(INPUT_PATCHES_PATH + '/*')
	
	# print 'filelist,', filelist
	for filename in filelist:
		for label in settings.LABELS:
			
			if string.find(filename + '/', label) != -1:  # hy:return index if found, or -1 if not found
				
				OUTPUT_PATH = DATA_PATH + label
				
				if DEBUG:
					print 'If error occurs, possibly because some old file name errors, e.g. contain _ in the end'
				
				filelist_tmp = glob.glob(INPUT_PATCHES_PATH + label + '*')
				random.shuffle(filelist_tmp)
				
				# hy:use fixed additional amount of data
				# random_add = 1
				# random_add = 7*len(settings.LABELS)
				
				# hy: use proportional setting
				# random_add = int(len(filelist_tmp) * 0.70)
				
				filelist_add = filelist_tmp[0:random_add]
				
				# print len(filelist_tmp)
				# for file in filelist_tmp: #hy if all data should be used
				for file in filelist_add:  # hy if only part data should be used
					file.split('/')[-1]
					# print file.split('/')[-1]
					original = cv2.imread(file)
					Height, Width, Channel = original.shape
					# print original.shape
					
					### Rotation start ##############################################################
					if Anti_clockwise == 1 and Rotation_Angle <> 0:
						# Counter-Clockwise: Zooming in, rotating, cropping
						# rotated_tmp = cv2.resize(original, (Width + 40, Height + 20), interpolation=cv2.INTER_LINEAR)
						# rotated_tmp = imutils.rotate(rotated_tmp, angle=Rotation_Angle)
						# rotated = rotated_tmp[10:Height + 10, 20:Width + 20]
						# print rotated.shape
						# rotated = imutils.resize(rotated, width=Width,
						#        height=Height)
						
						rotated = imutils.rotate(original, angle=Rotation_Angle)
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_st' + str(
							step) + '_rotatedCC_' + str(
							Rotation_Angle) + ImageType  # hy: split by .jpg, with '.' to avoid extra '.' in file name
						cv2.imwrite(new_file, rotated)
					
					if Clockwise == 1 and Rotation_Angle <> 0:
						# Clockwise
						rotated_tmp = cv2.resize(original, (Width + 40, Height + 20), interpolation=cv2.INTER_LINEAR)
						rotated_tmp = imutils.rotate(rotated_tmp, angle=Rotation_Angle * -1)
						rotated = rotated_tmp[10:Height + 10, 20:Width + 20]
						# print rotated.shape
						rotated = imutils.resize(rotated, width=Width, height=Height)
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_st' + str(
							step) + '_rotatedC_' + str(Rotation_Angle) + ImageType
						cv2.imwrite(new_file, rotated)
					
					### Rotation end   ##############################################################
					
					
					### Flipping begin ##############################################################
					if Flip_X == 1:
						flipped = cv2.flip(original, 0)
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_flippedX' + ImageType
						cv2.imwrite(new_file, flipped)
					
					if Flip_Y == 1:
						flipped = cv2.flip(original, 1)
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_flippedY' + ImageType
						cv2.imwrite(new_file, flipped)
					
					if Flip_XY == 1:
						flipped = cv2.flip(original, -1)
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_flippedXY' + ImageType
						cv2.imwrite(new_file, flipped)
					
					### Flipping end    ##############################################################
					
					
					### Perspective Transform begin  #################################################
					f_outs = []
					if do_perspectiveTransform_in == 1:
						# print 'file',file
						# h, w, ch = original.shape
						# print 'h,w,ch', h, w, ch
						# rand1 = randint(2,30)
						# rand2 = randint(2,30)
						aspect_w = int(Aspect_Factor * Width)
						for i in xrange(aspect_w, aspect_w + 2):
							for j in xrange(14, 16):
								pts1 = np.float32([[i, 0], [Width - i, 0], [j, Height - j], [Width - i, Height - j]])
								# pts1 = np.float32([[rand1, 0], [patch_size+rand1, 0], [rand2, patch_size], [patch_size, patch_size]])
								pts2 = np.float32([[0, 0], [Width, 0], [0, Height], [Width, Height]])  # leftT,rT,leftB,rB
								# pts2 = np.float32([[0, 0], [patch_size, 0], [0, patch_size], [patch_size, patch_size]])  # leftT,rT,leftB,rB
								
								M = cv2.getPerspectiveTransform(pts1, pts2)
								
								dst = cv2.warpPerspective(original, M, (Width, Height))  # (w,h)
								# dst = cv2.warpPerspective(img,M,(patch_size,patch_size)) #(w,h)
								f_out = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_persp_' + str(i) + '_' + str(
									j) + ImageType
								# f_out = '../Data/data_1_mix/save/prep/' + os.path.basename(file) + '_persp_' + str(i) + '_' + str(j) + ImageType
								print f_out
								f_outs.append(f_out)
								cv2.imwrite(f_out, dst)
						print 'can generate num of new files with perspective transformation', len(f_outs)
					
					### Perspective Transform end    #################################################
					
					
					### add noise begin ##############################################################
					if noise_level <> 0:
						img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
						
						# img = img.astype(np.float32)
						img_noised = img_gray + np.random.rand(Width, Height) * noise_level
						img_noised = (img_noised / np.max(img_noised)) * 255
						new_file = OUTPUT_PATH + os.path.basename(os.path.normpath(file))[:-4] + '_st' + str(
							step) + '_NOI' + ImageType
						cv2.imwrite(new_file, img_noised)
					### add noise end    ##############################################################


if do_rotateflipImg == 1:
	# rotateflipImg()
	rotateflipImg(Rotation_Angle=Rotation_Angle, Flip_X=Flip_X, noise_level=noise_level, step=step)

########################################################
# config for CROPPING and REMOVE SMALL AREA
# crop_source_path = '../tmp/distor_in/CAD/resized_all_sizes/'  # settings.LABELPATH
crop_source_path = '../tmp/distor_in/single/'  # settings.LABELPATH
crop_dest_path = '../tmp/resized/rest5/'
stride = 1  #
crop_method = 'seq'  # 'random', #'seq', #'random_dense'


def do_cropImg(crop_method=crop_method, crop_source_path=crop_source_path, crop_dest_path=crop_dest_path,
               patch_size=patch_size, patch_size_h=patch_size_h, stride=1):  # do_crop_Image
	print 'doing cropping'
	# dirs_complete = os.listdir(crop_source_path)
	dirs = os.listdir(crop_source_path)
	print 'crop source path', crop_source_path
	# dirs = [s for s in dirs_complete if os.path.basename(os.path.normpath(crop_dest_path)) in s and ImageType in s]
	# print dirs
	num_of_samples = 0
	bboxes_d = []
	bboxes_u = []
	for item in dirs:
		item = 'bg_office2.jpg'
		print crop_source_path + item
		if os.path.isfile(crop_source_path + item):
			
			im = Image.open(crop_source_path + item)
			fname_wo_end = os.path.splitext(item)[0]
			
			w, h = im.size
			# patch_size = int(min(w,h)*0.98)
			# patch_size_h = patch_size
			# print 'w,h:',w,',',h
			# hy usage: crop((left_x1,top_y1,   right_x2,bottom_y2)).save(file_path,file_extention,quality=0_to-100)
			# rangex = w - stride + 1
			# rangey = h - stride + 1
			do_cropImg_1_to_n(crop_method=crop_method, crop_source=crop_source_path + item, crop_dest_path=crop_dest_path,
			                  patch_size=patch_size, patch_size_h=patch_size_h, stride=1, crop_num=1, save_file=False)
		
		else:
			print 'End of file in this folder'
	
	num_of_samples = len(os.listdir(crop_dest_path))
	print 'files generated', num_of_samples
	
	if not os.listdir(crop_dest_path):
		print 'no file generated, check error'
	else:
		print 'files saved in', crop_dest_path


if do_crop_Image == 1:
	do_cropImg()


# do_cropImg(crop_method=crop_method,crop_source_path=crop_source_path,
#      crop_dest_path=crop_dest_path,patch_size=patch_size,patch_size_h=patch_size_h,stride=stride)


def do_cropImg_1_to_n(crop_method='', crop_source='', crop_dest_path='',
                      patch_size=0, patch_size_h=0, stride=0, crop_num=1, save_file=False):  # do_crop_Image
	print 'doing cropping', crop_method, crop_source, crop_dest_path, \
		patch_size, patch_size_h, stride, save_file
	
	bboxes_d = []
	bboxes_u = []
	
	if os.path.isfile(crop_source):
		
		im = Image.open(crop_source)
		fname_wo_end = os.path.splitext(os.path.basename(crop_source))[0]
		
		w, h = im.size
		print 'image original size w,h:', w, ',', h
		
		if crop_method == 'seq':
			
			# from lt to rb
			num_of_samples = 0
			for x in xrange(w):
				if (x * stride + patch_size) > w or num_of_samples > int(crop_num / 2):
					break
				for y in xrange(h):
					if (y * stride + patch_size) > h or num_of_samples > int(crop_num / 2):
						break
					print 'box', x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h
					# crop from left
					bbox = (x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h)
					bbox_im = im.crop(bbox)
					bboxes_u.append(bbox_im)
					
					bboxes_d.append(bbox)
					# save
					if save_file:
						slice = im.crop(bbox).save(crop_dest_path + fname_wo_end + '_d' + str(x * stride) + '_' + str(y * stride) +
						                           str(patch_size) + ImageType, optimize=True, bits=6)
					num_of_samples = len(bboxes_u) + len(bboxes_d)
			print 'can generate d-num of samples', num_of_samples
			
			# from rb to lt
			num_of_samples = 0
			for x in xrange(w):
				if (x * stride + patch_size) > w - x * stride or num_of_samples > crop_num:
					break
				for y in xrange(h):
					if (y * stride + patch_size) > h or num_of_samples > crop_num:
						break
					# crop from right
					bbox = (w - x * stride - patch_size, h - y * stride - patch_size_h, w - x * stride, h - y * stride)
					bbox_im = im.crop(bbox)
					bboxes_u.append(bbox_im)
					
					if save_file:
						slice = im.crop(bbox).save(
							crop_dest_path + fname_wo_end + '_u' + str(w - x * stride) + '_' + str(h - y * stride) +
							str(patch_size) + ImageType, optimize=True, bits=6)
					num_of_samples = len(bboxes_u) + len(bboxes_d)
			print 'can generate u-num of samples', num_of_samples
		
		if crop_method == 'random':  # '''
			# hy: random cropping, setting random left position
			for imgN in xrange(crop_num):
				x = randint(1, w - stride + 1)
				y = randint(1, h - stride + 1)
				bbox = (x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h)
				bbox_im = im.crop(bbox)
				bboxes_u.append(bbox_im)
				
				if save_file:
					slice = im.crop(bbox).save(
						crop_dest_path + fname_wo_end + str(imgN) + '_' + str(stride) + '_l' + str(x * stride) + '_' + str(
							y * stride) + ImageType, optimize=True, bits=6)
			
			# hy: random cropping, setting random right position
			for imgM in xrange(crop_num):
				x = randint(1, w - stride + 1)
				y = randint(1, h - stride + 1)
				bbox = (x * stride - patch_size, y * stride - patch_size_h, x * stride, y * stride)
				bbox_im = im.crop(bbox)
				bboxes_u.append(bbox_im)
				
				if save_file:
					slice = im.crop(bbox).save(
						crop_dest_path + fname_wo_end + str(imgM) + '_' + str(stride) + '_r' + str(x * stride) + '_' + str(
							y * stride) + ImageType, optimize=True, bits=6)
		
		if crop_method == 'random_dense':
			print 'method: random_dense'
			# patch_size = int(min(w, h)*0.995)
			# patch_size_h = patch_size
			print 'patch size', patch_size, ',', patch_size_h
			
			# hy: random cropping, setting random left position
			for imgN in xrange(crop_num):
				x = randint(1, w - stride - patch_size + 1)
				y = randint(1, h - stride - patch_size_h + 1)  # 346-1-295+1=50
				print x, ',', y
				bbox = (x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h)
				bbox_im = im.crop(bbox)
				print 'add image'
				bboxes_u.append(bbox_im)
				
				if save_file:
					slice = im.crop(bbox).save(
						crop_dest_path + fname_wo_end + '_' + str(stride) + '_ex' + str(
							x * stride) + '_' + str(y * stride) + ImageType, optimize=True, bits=6)
			
			# hy: random cropping, setting random right position
			for imgM in xrange(crop_num):
				x = randint(1 + patch_size, w - 1)
				y = randint(1 + patch_size_h, h - 1)  # 1+224, 234
				bbox = (x * stride - patch_size, y * stride - patch_size_h, x * stride, y * stride)
				bbox_im = im.crop(bbox)
				bboxes_u.append(bbox_im)
				print 'add image'
				
				if save_file:
					slice = im.crop(bbox).save(
						crop_dest_path + fname_wo_end + '_' + str(stride) + '_ex' + str(
							x * stride - patch_size) + '_' + str(y * stride - patch_size_h) + ImageType, optimize=True, bits=6)
			
			if max(w, h) / min(w, h) >= 1.5:
				# hy: random cropping, setting random center position
				print 'rectangular img'
				for imgN in xrange(crop_num):
					x = randint(int((w - patch_size) / 5), w - stride - patch_size + 1)
					y = randint(1, h - stride - patch_size_h + 1)  # 235-1-224+1=11
					# y = randint(1, h - stride - patch_size_h + 1)
					bbox = (x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h)
					bbox_im = im.crop(bbox)
					print 'add image'
					bboxes_d.append(bbox_im)
					
					if save_file:
						slice = im.crop(bbox).save(
							crop_dest_path + fname_wo_end + '_' + str(stride) + '_ex' + str(
								x * stride) + '_' + str(y * stride) + ImageType, optimize=True, bits=6)
		
		else:
			print 'End of file in this folder'
	
	num_of_samples = len(os.listdir(crop_dest_path))
	print 'files generated', num_of_samples
	
	if not os.listdir(crop_dest_path):
		print 'no file generated, check error'
	else:
		print 'files saved in', crop_dest_path
	return bboxes_u, bboxes_d


# First crop center, then resize to patch size
def crop_center(f_img=None, crop_num=1, crop_w=0, crop_h=0, crop_dest_path='', save_file=False):
	print 'crop center from img', f_img
	im = Image.open(f_img)
	# im = f_img
	w, h = im.size  # correct order w,h
	
	patch_size = int(min(h, w) * 1)
	patch_size_h = patch_size
	print 'h,w', h, w
	bboxes = []
	
	for imgN in xrange(crop_num):
		# hy: random cropping, setting random center position
		x = randint(int((w - patch_size) / 2 * 0.99),
		            int((w - patch_size) / 2 * 1.07))  # adjust obj position in resulting image
		y = randint(int((h - patch_size) / 2 * 0.99), int((h - patch_size) / 2 * 1.01))
		# x = int((w - patch_size)/2 *0.97)
		# y = int((h - patch_size)/2 *0.9)
		bbox = (x * stride, y * stride, x * stride + patch_size, y * stride + patch_size_h)
		bbox_im = im.crop(bbox)
		bbox_im.thumbnail((crop_h, crop_w), Image.ANTIALIAS)  # an action, check order
		bbox_res1 = bbox_im.convert("RGBA")
		bboxes.append(bbox_res1)
		
		if save_file:
			fbase_name = os.path.basename(f_img)
			fname_wo_end = os.path.splitext(fbase_name)[0]
			print fname_wo_end, 'f full', crop_dest_path + fname_wo_end + '_' + str(stride) + '_ex' + str(
				x * stride) + '_' + str(y * stride) + ImageType
			slice = im.crop(bbox).save(
				crop_dest_path + fname_wo_end + '_' + str(stride) + '_ex' + str(x * stride) + '_' + str(y * stride) + ImageType,
				optimize=True, bits=6)
			print 'file saved in', crop_dest_path
	
	return bboxes


######################################################
crop2_source_path = '../tmp/distor_in/camImgs/testPkg7/'  # settings.LABELPATH
crop2_dest_path = '../tmp/distor_in/camImgs/'
equally_crop = 0
select_crop = 1
crop_opt = 'equal'  # 'equal' #'custom'
num_of_parts_per_row = 4


def cropImage_none_overlap(crop2_source_path=crop2_source_path, crop2_dest_path=crop2_dest_path,
                           crop_opt=crop_opt, num_of_parts_per_row=num_of_parts_per_row):  # do_crop_none_overlap
	print 'doing cropping without overlap'
	if not os.listdir(crop2_source_path):
		print 'no file to crop'
		exit(1)
	# patch_size_h = 28  # int(patch_size * 0.75)
	dirs_complete = os.listdir(crop2_source_path)
	dirs = [s for s in dirs_complete if os.path.basename(os.path.normpath(crop_dest_path)) in s]
	# dirs = [s for s in dirs_complete if 'oben' in s]
	print dirs
	
	for item in dirs:
		if os.path.isfile(crop2_source_path + item):
			# item = 'hinten_ori1_rz400.jpg'
			im = Image.open(crop2_source_path + item)
			
			# hy: two methods for getting file name
			# fpath, fname = os.path.split(source_path + item)
			# fname =  os.path.basename(source_path + item)
			
			fname_wo_end = os.path.splitext(item)[0]
			# print 'image name:', fname_wo_end
			
			w, h = im.size
			# print 'w,h:',w,',',h
			# hy usage: crop((left_x1,top_y1,   right_x2,bottom_y2)).save(file_path,file_extention,quality=0_to-100)
			# hy: split image into parts
			if crop_opt == 'equal':
				# 10x10 parts
				for row in xrange(0, num_of_parts_per_row):
					for col in xrange(0, num_of_parts_per_row):
						bbox = (int(col * w / num_of_parts_per_row), int(row * h / num_of_parts_per_row),
						        int((col + 1) * w / num_of_parts_per_row), int((row + 1) * h / num_of_parts_per_row))
						slice = im.crop(bbox).save(crop2_dest_path + fname_wo_end + 'r_' + str(row) + '_c_' + str(col) + ImageType,
						                           optimize=True, bits=6)
						print 'files saved in', crop2_dest_path, len(os.listdir(crop2_dest_path))
			if crop_opt == 'custom':
				# left, right parts
				bbox = (int(0.1 * w), int(0.24 * h), int(0.8 * w), int(0.85 * h))
				slice = im.crop(bbox).save(crop2_dest_path + fname_wo_end + '_' + ImageType, optimize=True, bits=6)
			
			# bbox = (int(0.5 * w), 0, w, h,)
			# slice = im.crop(bbox).save(crop2_dest_path + fname_wo_end + '_tou' + ImageType, optimize=True, bits=6)


if do_cropImage_none_overlap == 1:
	cropImage_none_overlap()

scalesize = 320  # 48 for resize scale 172, 79,
resize_INPUT_PATH = '../Data/data_3_unet/'  # settings.LABELPATH
# resize_INPUT_PATH = '../tmp/input_patches/' #settings.LABELPATH
resized_path = '../Data/data_3_unet/resized/'


def resizeImage(resize_INPUT_PATH=resize_INPUT_PATH, resized_path=resized_path, scalesize=scalesize):  # do_resizeImg
	print 'doing resize'
	# resize_INPUT_PATH = '../Test_Images/'
	
	dirs = os.listdir(resize_INPUT_PATH)
	for item in dirs:
		# print 'yes, path is correct'
		if os.path.isfile(resize_INPUT_PATH + item):
			# print 'yes, the folder contains files'
			# item = 'hinten_ori1.jpg' #in case for specified  one image
			im = Image.open(resize_INPUT_PATH + item)
			w, h = im.size  # hy corrected order of w,h
			print 'w,h:', w, ',', h
			if w > h:
				scalesize_b = scalesize
				scalesize_a = int(scalesize * w / h)  # hy keep proportion
			else:
				scalesize_a = scalesize
				scalesize_b = scalesize * int(h / w)
			# fpath, fname = os.path.split(resize_INPUT_PATH + item)
			fname = os.path.basename(resize_INPUT_PATH + item)
			f_out = resized_path + fname[0:-4] + '_rz' + str(scalesize) + ImageType
			print 'f_out', f_out
			imResize = im.resize((scalesize_a, scalesize_b),
			                     Image.BICUBIC)  # hy resize(w,h) corrected order. todo try bicubic, instead ANTIALIAS
			imResize.save(f_out)  # hy: use this no restriction for image type, jpg,jpeg,png
		# imResize.save(f_out, ImageType[1:], quality=90) #hy: image type do not include '.'
	
	if not os.listdir(resized_path):
		print 'no file generated, check error'
	
	# img = img.resize((new_width, new_height), Image.ANTIALIAS)
	# img.save('output image name.png')  # format may what u want ,*.png,*jpg,*.gif


if do_resizeImg == 1:
	resizeImage()


def resizeCVimgTest(im, h, w):
	basename = os.path.basename(im)
	print basename
	im = cv2.imread(im)
	im = cv2.resize(np.uint8(im), (h, w))
	cv2.imwrite('../Data/data_2/' + basename, im)


# resizeCVimgTest('../tmp/distor_in/links_ww7.jpg',320,320)
# frame_crop_resize_gray = imutils.resize(im, width=settings.w_resize)
def resizeCVimg(im, h, w, fileame=None):
	# basename = os.path.basename(im)
	# print basename
	im = cv2.resize(np.uint8(im), (h, w))
	# cv2.imwrite('../Data/data_2/'+basename,im)
	return im


def augment_data_1_to_n(im, h, w, save_file=False):
	name = os.path.basename(im)[:-len(ImageType)]
	save_path = '../Data/data_2/'
	print 'augment data', name
	im = cv2.imread(im)
	im_proportions = []
	ims_resize = []
	
	# first cut different proportion from whole image
	def cutCVimg(im, ori_h, ori_w, ratio):
		w = int(ori_w * ratio * 0.95)  # simulate different detection area sizes
		h = int(ori_h * ratio)
		im = im[ori_h - h:h, ori_w - w:w]  # y1:y2,x1:x2
		return im
	
	for r in xrange(7, 9):
		im = cutCVimg(im, im.shape[0], im.shape[1], r * 0.125)  # for rectangular use 0.12
		im_proportions.append(im)
		cv2.imshow('pro', im)
		cv2.waitKey(10)
	
	# second, resize them to seg output size hxw
	for im, i in zip(im_proportions, xrange(len(im_proportions))):
		im = resizeCVimg(im, 320, 320)
		ims_resize.append(im)
		cv2.imshow('resize', im)
		cv2.waitKey()
		if save_file:
			cv2.imwrite(save_path + name + '_' + str(i) + ImageType, im)
	
	# third, add all flipping
	# for im in ims_resize:
	for im, i in zip(ims_resize, xrange(len(ims_resize))):
		for flipType in xrange(-1, 2):
			im = cv2.flip(im, flipType)
			filename = save_path + name + '_' + str(i) + '_flip' + str(flipType + 1) + ImageType
			print 'filename', filename
			cv2.imshow('flip' + str(i) + str(flipType + 1), im)
			cv2.waitKey()
			if save_file:
				cv2.imwrite(filename, im)
	
	if save_file:
		print '\nfiles saved in', save_path

if do_augment_data:
	augment_data_1_to_n('../tmp/distor_in/vorn_ww3.jpg', 320, 320, save_file=True)

#########################################################################
if prepare_active_fields == 1:
	print 'prepare active fields'
	
	crop_source_path = '../Test_Images/test_active_fields/'
	crop_dest_path = '../Test_Images/test_active_fields/vorn/'  # MUST use label folder here to fit following sequence
	
	crop_method = 'random_dense'
	# do_cropImg(crop_method=crop_method, crop_source_path=crop_source_path,crop_dest_path=crop_dest_path,stride=1)
	
	tmp_path = crop_source_path + 'tmp/'
	cmd = 'mv ' + crop_dest_path + '*.jpg ' + tmp_path
	# os.system(cmd)
	
	# resize
	# resizeImg(resize_INPUT_PATH=crop2_source_path, resized_path=crop2_source_path,scalesize=400)
	
	# crop2_dest_path = crop_dest_path
	# select an optimal image then crop
	cropImage_none_overlap(crop2_source_path=tmp_path, crop2_dest_path=crop_dest_path, crop_opt='equal',
	                       num_of_parts_per_row=4)


#########################################################################
# hy: config for resizeImg()
# !/usr/bin/python
# from PIL import Image
# import os, sys

#   os.path.dirname(a_file_w_path) --- output path
# 1 != 1 # false
# 1 <> 1 # false
# [] is [] # false (distinct objects)
# a = b = []; a is b # true (same object)

# check folder contains filename with pattern
# for f in os.listdir(OUTPUT_PATH +'/'):
# if re.search('/*flipped*.*', f):


######################################################################




def add_background_to_CAD():  # add_background
	create_full_black_mask = True
	
	# Two images are needed: one CAD training image and one background image
	bg_crop_source = '../tmp/distor_in/input/bg_office1.jpg'
	if create_full_black_mask:
		CAD_lila_img_f = bg_crop_source
	else:
		CAD_lila_img_f = '../tmp/distor_in/CAD/CAD_lila_background/vorn_lila_background.jpg'
	# links3_lila_background.png
	# vorn2_lila_background.jpg  links1_lila_background.jpg
	dest_path = '../Data/data_3_u_net/resized/cad2/'
	print 'add background'
	print 'prepare background images by cropping one image'
	crop_dest_path = '../tmp/resized/rest4/'
	
	bgs, bgs_d = do_cropImg_1_to_n(crop_method='random_dense', crop_source=bg_crop_source, crop_dest_path=crop_dest_path, \
	                               patch_size=320, patch_size_h=320, stride=1, crop_num=4, save_file=False)
	bg = bgs[0]
	print 'bgs len', len(bgs), bg
	
	border_s = 0
	border_l = 1
	
	if border_s == 1:  #####################################################################################################
		# for vorn and hinten, Crop center part of the CAD image with lila background, get square form CAD image with lila background
		CAD_lila_imgs = crop_center(f_img=CAD_lila_img_f, crop_num=2, crop_w=320, crop_h=320, crop_dest_path=crop_dest_path,
		                            save_file=False)
		new_im = CAD_lila_imgs[0]
	
	#####################################################################################################
	if border_l == 1:
		# for others, resize CAD_lila
		CAD_lila_img_r = Image.open(CAD_lila_img_f)
		w, h = CAD_lila_img_r.size
		print '***** w,h', w, h, 'h/w', h / w * 316
		old_size = (316, 172)  # (316,172)
		CAD_lila_img_r.thumbnail(old_size, Image.ANTIALIAS)  # an action
		print 'size', CAD_lila_img_r.size
		new_size = (320, 320)
		new_im = Image.new("RGB", new_size)  ## default black!
		
		new_im.paste((80, 80, 192), (0, 0, 320, 320))  ##
		
		new_im.paste(CAD_lila_img_r, (int((new_size[0] - old_size[0]) / 2),
		                              int((new_size[1] - old_size[1]) / 2)))
		new_im.show()
	
	#####################################################################################################
	CAD_lila_img = new_im.convert("RGBA")
	
	# debug
	# print 'save test image in', dest_path
	# CAD_lila_img.save(dest_path + 'test.jpg')
	
	datas_CAD = CAD_lila_img.getdata()  # new
	# debug
	# print 'datas-obj', datas_CAD
	
	if not create_full_black_mask:
		for item_bg, i in zip(bgs, range(42, 44)):  # i: index in image names, depending on number of bgs
			# item_bg.thumbnail((h, w), Image.ANTIALIAS)
			item_bg = item_bg.convert("RGBA")
			item_bg_copy = item_bg.copy()
			datas_bg = item_bg.getdata()
			# debug
			# print 'datas-bg', datas_bg
			
			newData = []
			for data_obj, data_bg in zip(datas_CAD, datas_bg):
				if data_obj[0] >= 20 and data_obj[0] <= 106 and \
												data_obj[1] >= 30 and data_obj[1] <= 116 and \
												data_obj[2] >= 125 and data_obj[2] <= 245:
					newData.append((data_bg[0], data_bg[1], data_bg[2]))
				else:
					newData.append(data_obj)
			
			item_bg.putdata(newData)
			item_bg.save(dest_path + 'cad' + '_%03d.jpg' % i)
			print 'files saved in', dest_path
			
			########
			newData = []
			for data_obj, data_bg in zip(datas_CAD, datas_bg):
				if data_obj[0] >= 20 and data_obj[0] <= 106 and \
												data_obj[1] >= 30 and data_obj[1] <= 116 and \
												data_obj[2] >= 125 and data_obj[2] <= 245:
					newData.append((0, 0, 0))
				else:
					newData.append((255, 255, 255))
			
			item_bg_copy.putdata(newData)
			item_bg_copy.save(dest_path + 'cad' + '_m' + '_%03d.jpg' % i)
			print 'files saved in', dest_path
		########
	if create_full_black_mask:
		new_size = (320, 320)
		new_im = Image.new("RGB", new_size)  ## default black!
		new_im.save(dest_path + 'cad_m_04000.jpg')
		bg.save(dest_path + 'cad_04000.jpg')

if add_background == 1:
	add_background_to_CAD()


########################################################
def remove_small_obj():  # do_remove_small_obj
	print 'doing remove images with small object'
	count = 0
	folders = []
	# if use test image
	
	# dirs = os.listdir(crop_dest_path+settings.LABELS[1])
	dirs = os.listdir(crop_dest_path)
	ori_num_of_files = len(dirs)
	print crop_dest_path
	if not dirs:
		print 'empty directory'
		exit(0)
	
	for item in dirs:  # use defined directory
		if os.path.isfile(crop_dest_path + item):
			im = Image.open(crop_dest_path + item)
			im = im.convert('LA')  # convert Image object image to gray
			
			# hy: two methods for getting file name
			# fpath, fname = os.path.split(source_path + item)
			# fname =  os.path.basename(source_path + item)
			
			# fname_wo_end = os.path.splitext(item)[0]
			# print 'image name:', fname_wo_end
			
			w, h = im.size  # hy corrected order of w,h
			# print 'w,h:',w,',',h
			# print 'pix',pix, ',', pix2
			
			if h <> w:
				# if h <> patch_size or w <> patch_size:
				print 'some images are not defined. w,h:', w, ',', h
				print 'Make sure the size is uniformed or comment out this condition.', item
			# os.remove(crop_dest_path+item)
			
			whole_area = h * w
			no_obj_area = 0
			
			# method 1 to get pixel
			pix = im.load()  # get pixel values of Image object image, get point pixel use e.g. print pix[x,y]
			# print 'max pixel', im.getextrema()
			
			for y in xrange(h):
				for x in xrange(w):
					# method 1
					# pix = pix[x,y]   #get pixel value of point x,y
					
					# method 2
					# pix = im.getpixel((x,y))
					# print 'pix of test image', pix[x,y]
					
					# print 'pix of test image',pix[x,y]
					
					# hy normally do not use two bound threshold
					# threshold_color_min=255
					# threshold_color_max=0
					
					# threshold_min = (threshold_color_min,255)#for removing white compare with (thresholdvalue,255)
					# threshold_max = (threshold_color_max,0) # set (250,255) to (255,255) to remove area close to white
					# print pix[x,y]
					if pix[x, y] == (0, 255):
						# if pix[x,y] >= threshold_min and pix[x,y] <= threshold_max:
						no_obj_area = no_obj_area + 1
					# print 'pix backg', pix[x,y] #pix[x,y]
			
			proportion = float(no_obj_area / whole_area)
			# print 'no object area', no_obj_area,'proportion', proportion
			if proportion > 0.38:
				count = count + 1
				print 'prop', proportion, ',count to remove', count, ',', crop_dest_path + item
				cmd = 'mv ' + crop_dest_path + item + ' ../Data/data_1_mix/'
				os.system(cmd)  # os.remove(crop_dest_path+item)
	print '\nremaining files:', ori_num_of_files, '-', count, '=', ori_num_of_files - count  # , len(os.listdir(crop_dest_path))

if do_remove_small_obj == 1:
	remove_small_obj()


######################
def convert_PCA():
	pca_source_path = '../tmp/input_patches/'  # settings.LABELPATH
	pca_dest_path = '../Data/'
	print 'doing convert PCA'
	
	dirs = os.listdir(pca_source_path)
	if not dirs:
		print 'empty directory'
		exit(0)
	for item in dirs:  # use defined directory
		if os.path.isfile(pca_source_path + item):
			im = Image.open(pca_source_path + item)
			
			w, h = im.size  # hy corrected order of w,h
			# print 'w,h:',w,',',h
			fpath, fname = os.path.split(pca_source_path + item)
			fname = os.path.basename(pca_source_path + item)
			print 'fpath, fname', fpath, '---', fname
			
			gray_path = pca_dest_path  # hy: here todo optimize the counter for character position
			print 'gray_path', gray_path
			f_out = gray_path + fname[0:-4] + '_cvt' + ImageType
			print 'f_out', f_out
			im.save(f_out)  # hy: use this no restriction for image type, jpg,jpeg,png

if do_convertPCA == 1:
	convert_PCA()


########################################################
def create_test_slices(img, patch_size, label):
	im = img
	# h_b, w_b = im.shape
	h_b, w_b = im.size
	print 'read test image ok', h_b, ', ', w_b
	bboxes = []
	slices = ['../tmp00.png', '../tmp01.png', '../tmp10.png',
	          '../tmp11.png', '../tmp_c.png']
	
	# corner (0,0)
	# bbox1 = im[1:1 + patch_size, 1:1 + patch_size] #crop method for numpy array object, cv2 image, which has attribute shape
	bbox1 = (1, 1, 1 + patch_size, 1 + patch_size)  # crop method for Image object which has attribute size
	
	# corner (0,1)   1 + patch_size
	# bbox2 = im[1:1 + patch_size, w_b - patch_size: w_b]
	bbox2 = (1, w_b - patch_size, 1 + patch_size, w_b)
	
	# corner (1,0)
	# bbox3 = im[h_b - patch_size:h_b, 1: patch_size +1]
	bbox3 = (h_b - patch_size, 1, h_b, patch_size + 1)
	
	# corner (1,1)
	# bbox4 = im[h_b - patch_size:h_b,  w_b - patch_size: w_b]
	bbox4 = (h_b - patch_size, w_b - patch_size, h_b, w_b)
	
	# center (c,c)
	# bbox5 = im[int(h_b/2 - patch_size/2)-1: int(h_b/2 + patch_size/2), int(w_b/2 - patch_size/2)-1: int(w_b/2 + patch_size/2)]
	bbox5 = (int(h_b / 2 - patch_size / 2) - 1, int(w_b / 2 - patch_size / 2) - 1, int(h_b / 2 + patch_size / 2),
	         int(w_b / 2 + patch_size / 2))
	
	bboxes.append(bbox1)
	bboxes.append(bbox2)
	bboxes.append(bbox3)
	bboxes.append(bbox4)
	bboxes.append(bbox5)
	bboxes_len = len(bboxes)
	
	slices_files = []
	folderstr = settings.LABELS[int(label)] + '/'
	for boxindex in range(0, bboxes_len, 1):
		slice = im.crop(bboxes[boxindex]).save(slices[boxindex], optimize=True, bits=6)
		# slice = bboxes[boxindex] #for numpy.ndarray object
		
		cmd = 'mv ' + slices[boxindex] + ' ' + settings.tmp + folderstr  # hy: move
		os.system(cmd)
		slice_pathname = settings.tmp + folderstr + slices[boxindex].split('/')[1]
		slices_files.append(slice_pathname)
	# print 'slice file path', slices_files[boxindex]
	print 'slices created ok'
	return slices_files


"""
 #changing aspect ratio slightly

 new_Width = np.int(Width*Aspect_Factor)
 deformed = cv2.resize(original,(new_Width,Height))
 new_file = file.split('jpg')[-2]+'_deformed_.jpg'
 cv2.imwrite(new_file,deformed)

 #cropping
 #center
 cropped = original[Crop_h:Height-Crop_h,Crop_w:Width-Crop_w]
 new_file = file.split('jpg')[-2]+'_CroppedC_.jpg'
 cv2.imwrite(new_file,cropped)
 #leftUp
 cropped = original[:Height-Crop_h,:Width-Crop_w]
 new_file = file.split('jpg')[-2]+'_CroppedLU_.jpg'
 cv2.imwrite(new_file,cropped)
 #LeftDown
 cropped = original[Crop_h:,:Width-Crop_w]
 new_file = file.split('jpg')[-2]+'_CroppedLD_.jpg'
 cv2.imwrite(new_file,cropped)
 #RightUP
 cropped = original[:Height-Crop_h,Crop_w]
 new_file = file.split('jpg')[-2]+'_CroppedRU_.jpg'
 cv2.imwrite(new_file,cropped)
 #RightDown
 cropped = original[Crop_h:,:Width-Crop_w]
 new_file = file.split('jpg')[-2]+'_CroppedRD_.jpg'
 cv2.imwrite(new_file,cropped)
 #CenterExtreme
 #cropped = original[Crop_h*2:Height-Crop_h*2, Crop_w*3, Width-Crop_w*3] #hy:be careful not out of range
 ropped = original[Crop_h * 2:Height - Crop_h * 2, Crop_w * 3 : Width - Crop_w * 3]
 new_file = file.split('jpg')[-2]+'_CroppedCExtreme_.jpg'
 cv2.imwrite(new_file,cropped)

 """
if REMOVE_Inputpatches == 1:
	print 'removing patches subfolders data'
	for i in xrange(length):
		folder_to_empty = settings.patches + settings.LABELS[i]
		cmd = 'rm -r ' + folder_to_empty + '*'  # hy: remove recursively
		os.system(cmd)

if REMOVE_Inputpatches_outside == 1:
	print 'removing patches outside data'
	cmd = 'rm -r ' + settings.patches + '*' + ImageType  # hy: remove recursively
	os.system(cmd)

if REMOVE_resized == 1:
	print 'removing resized Data'
	cmd = 'rm -r ' + settings.resized + '*.jpg'  # hy: remove recursively
	os.system(cmd)



if REMOVE_Data == 1:
	print '\nremoving data'
	for i in xrange(length):
		folder_to_empty = settings.data + settings.LABELS[i]
		cmd = 'rm -r ' + folder_to_empty + '*'  # hy: remove recursively
		os.system(cmd)

if REMOVE_Test_Images == 1:
	print 'removing test images'
	for i in xrange(length):
		cmd = 'rm -r ' + settings.test_images + settings.LABELS[i] + '/*.*'  # hy: remove recursively
		os.system(cmd)

if split_Data == 1:
	print 'splitting data'
	for i in xrange(length):
		# cmd = 'mv ' + '../Data/data_3/cam2/' + fileprefix[i] + '*.*' + ' ' + '../Data/data_3/' + folder_list[i]
		cmd = 'mv ' + settings.data + fileprefix[i] + '*.*' + ' ' + settings.data + settings.LABELS[i]
		os.system(cmd)

if split_Inputpatches == 1:
	print 'splitting input patches'
	for i in xrange(length):
		cmd = 'mv ' + settings.patches + fileprefix[i] + '*.* ' + settings.patches + settings.LABELS[i]
		os.system(cmd)

if split_Test_Images == 1:
	print 'splitting test images'
	for i in xrange(length):
		cmd = 'mv ' + settings.test_images + fileprefix[i] + '*.* ' + settings.test_images + settings.LABELS[i]
		os.system(cmd)

if move_from_tmp_to_patches == 1:
	print 'moving some data from tmp to patches'
	for i in xrange(length):
		cmd = 'mv ' + settings.tmp + settings.LABELS[i] + '/' + fileprefix[i] + '*.* ' + settings.patches + settings.LABELS[
			i]  # hy: cp to
		os.system(cmd)

if move_from_tmp2_to_patches == 1:
	print 'moving some data from tmp to patches'
	for i in xrange(length):
		# cmd = 'mv ' + tmp2 + fileprefix[i] + '*.* ' + test_images + folder_list[i]
		cmd = 'mv ' + settings.tmp2 + fileprefix[i] + '*.* ' + settings.patches + settings.LABELS[i]
		os.system(cmd)

if move_from_tmp2_to_testImg == 1:
	print 'moving from tmp2 to test images sub-folders'
	for i in xrange(length):
		# cmd = 'mv ' + data + folder_list[i]  + '/' + fileprefix[i] + '*.* ' + patches + folder_list[i]
		cmd = 'mv ' + settings.tmp2 + fileprefix[i] + '*.jpg ' + settings.test_images + settings.LABELS[i]
		os.system(cmd)

if copy_from_data_to_patches == 1:
	print 'copying from data to patches subfolder'
	for i in xrange(length):
		cmd = 'cp -r ' + settings.data + settings.LABELS[i] + '/' + fileprefix[i] + '*.jpg ' + settings.patches + \
		      settings.LABELS[i]  # hy: cp to
		# cmd = 'cp -r ' + patches + folder_list[i] + '/' + fileprefix[i] + '*.jpg ' + data + folder_list[i]
		os.system(cmd)

if copy_from_patches_to_testImg == 1:
	print 'copying from patches subfolder to testImages'
	for i in xrange(length):
		cmd = 'cp -r ' + settings.patches + settings.LABELS[i] + '/' + fileprefix[i] + '*.* ' + settings.test_images + \
		      settings.LABELS[i]  # hy: cp to
		os.system(cmd)

if copy_from_misclassifed_to_data == 1:
	print 'copying from misclassified to data outside'
	cmd = 'cp -r ' + settings.misclassifed + '*.* ' + settings.data
	os.system(cmd)

# Todo create config file
# Todo Add flags for different distortions
# Todo Remove some transforms before show
# Todo Work on a better rotation
