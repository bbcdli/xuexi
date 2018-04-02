#tools.py
#####################################################################################################
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageOps
import time
import random
import glob
from functools import wraps
from PIL import Image  # for resize func
from random import randint
import os
import sys
import datetime
#import settings  # hy: collection of global variables
import cv2
import numpy as np
import tensorflow as tf
from sklearn import datasets
import math
import settings
def SAVE_Images(filename, filepath):
    OUTPUT_PATH = filepath
    cmd = 'cp ' + filename + ' ' + OUTPUT_PATH
    os.system(cmd)
def SAVE_CorrectClassified_frame(name_str, img, save=False):
    if save:
        imgNum = len([name for name in os.listdir(settings.CorrectClassified) if
                      os.path.isfile(os.path.join(settings.CorrectClassified, name))])
        img_name = name_str
        # print 'num of files', misNum
        if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
            cv2.imwrite(img_name, img)
        else:
            print 'CorrectClassified Folder is full !!!'

def SAVE_Misclassified_Img(img, save=False):
    if save == 1:
        imgNum = len([name for name in os.listdir(settings.Misclassified) if
                      os.path.isfile(os.path.join(settings.Misclassified, name))])
        img_name = img
        if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
            SAVE_Images(img_name, settings.Misclassified)
        else:
            print 'Misclassified Folder is full !!!'

def SAVE_Misclassified_frame(name_str, img, save=False):
    if save == 1:
        imgNum = len([name for name in os.listdir(settings.Misclassified) if
                      os.path.isfile(os.path.join(settings.Misclassified, name))])
        img_name = name_str
        # print 'num of files', misNum
        if imgNum < settings.maxNumSaveFiles:  # hy: to avoid full disk error!
            cv2.imwrite(img_name, img)
        else:
            print 'Misclassified Folder is full !!!'
def get_ground_truth_label_im(label_text, default=False):
    target = 0
    if 'non' not in label_text:
        target = 0
    if 'non' in label_text:
        target = 1
    return target
def reduce_mean_stdev(images, print_val=False):
    mean = np.mean(images)
    stdev = np.std(images)
    if print_val:
        print 'mean %d,stdev %d', (mean, stdev)
    images = images - mean
    images_reduced_mean = images / stdev
    return images_reduced_mean

def import_data_k_segnet(im_path, label_path, file_imgs, file_masks, h, w, maxNum,
rnd_blurriness_min=0,rnd_blurriness_max=10,rnd_darkness_min=0 ,rnd_darkness_max=10,
                         do_Flipping=False, do_gblur=False, do_darken=False):
    from scipy import ndimage as nd
    import scipy.misc as misc
    print 'maxNum',maxNum
    INPUT_CH = 1
    rnd_flip, rnd_gblur, rnd_darken = [], [], []
   
    if do_Flipping:
        rnd_flip = random.sample(xrange(maxNum - 1), maxNum - 1)
    if do_gblur:
        rnd_gblur = random.sample(xrange(maxNum - 1), maxNum - 2)
    if do_darken:
        rnd_darken = random.sample(xrange(maxNum - 1), maxNum - 1)  # 80
    maxNum = 3 * len(rnd_flip) + len(rnd_gblur) + len(rnd_darken) + len(file_imgs) + 4
    # todo: check len of total input images
    #d = 0 if INPUT_CH == 1 else -1
    print 'maxNum after:', maxNum
    # out: images pixel value divided by 255, so they have value between 0 and 1,
    # #mask is composed of 0s and 1s, for two classes
    images = np.zeros((maxNum, 1, h, w))
    masks = np.zeros((maxNum, 1, h, w))
    print 'load data', im_path, h, w, maxNum, do_Flipping
    data_counter = 0
    num_raw_files = len(file_imgs)
    for i, img, m in zip(xrange(1, num_raw_files + 1), file_imgs, file_masks):
        # fimg = im
        fimg = im_path + img
        print '\n#', i, 'of', num_raw_files, 'img file name:', fimg
        fmask = label_path + m  #
        print '\n#', i, 'of', len(file_imgs), 'mask file name:', fmask
        img = cv2.imread(fimg, 0)  # d>0: 3-channel, =0: 1-channel, <0:no change
        mask = cv2.imread(fmask, 0)  # always greyscale
        if mask is None or img is None:
            continue
        img = cv2.resize(img, (h, w))
        mask = cv2.resize(mask, (h, w))
        img = np.float32(img.reshape(INPUT_CH, h, w))
        mask = mask.reshape(1, h, w)
        img = img / 255.0
        mask = mask / 255.0
        if do_Flipping and i in rnd_flip:
            img_copy = img.copy()
            mask_copy = mask.copy()
            for fl in range(-1, 2):
                flipped_img = cv2.flip(img_copy, fl)
                flipped_mask = cv2.flip(mask_copy, fl)
                images[data_counter, :, :, :] = flipped_img
                masks[data_counter, :, :, :] = np.float32(flipped_mask > 0)
                data_counter += 1
        if do_gblur and i in rnd_gblur:
            img_copy = img.copy()
            mask_copy = mask.copy()
            rnd_blurriness = 0.01 * randint(rnd_blurriness_min, rnd_blurriness_max)
            gblur_img = nd.gaussian_filter(img_copy, sigma=rnd_blurriness)
            images[data_counter, :, :, :] = gblur_img
            masks[data_counter, :, :, :] = np.float32(mask_copy > 0)
            # masks[data_counter, :, :, :] = np.expand_dims(np.array(mask_copy > threshold), 2).astype(
            # np.uint32)
            data_counter += 1
        if do_darken and i in rnd_darken:
            img_copy = cv2.imread(fimg)
            img_copy = cv2.resize(img_copy, (h, w))
            mask_copy = mask.copy()
            dark_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            dark_img = Image.fromarray(dark_img)  #
            rnd_darkness = 0.01 * randint(rnd_darkness_min, rnd_darkness_max)
            dark_img = dark_img.point(lambda p: p * rnd_darkness)
            dark_img = cv2.cvtColor(np.array(dark_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite('../tmp_k1.png', dark_img)
            if INPUT_CH == 1:
                dark_img = cv2.imread('../tmp_k1.png', 0)
            else:
                dark_img = cv2.imread('../tmp_k1.png')
            cmd = 'rm ../tmp_k1.png'
            os.system(cmd)
            # dark_img = cv2.resize(dark_img, (h, w))
            dark_img = np.float32(dark_img.reshape(INPUT_CH, h, w))
            dark_img = dark_img / 255.0
            images[data_counter, :, :, :] = dark_img
            masks[data_counter, :, :, :] = np.float32(mask_copy > 0)
            data_counter += 1
        images[data_counter, :, :, :] = img
        masks[data_counter, :, :, :] = np.float32(mask > 0)
        data_counter += 1
    print 'total', data_counter, 'images and', data_counter, 'masks are loaded'
    return images[0:data_counter, :, :, :], masks[0:data_counter, :, :, :]

def process_res(confMat1_TEST, confMat2_TEST, RES, frame_crop, SAVE_CorrectClassified=False, SAVE_Misclassified=False,
                im_i=1,
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

def count_diff_pixel_values(picture, h, w):
    ds = []
    for row in xrange(h):
        for col in xrange(w):
            if picture[row][col] not in ds:
                ds.append(picture[row][col])
    return len(ds), ds

def print_label_title():
    titles = ['empty','non-empty']
    print titles
def calc_dice_simi(seg, gt, img_name, k=1):
    # segmentation
    # seg = np.zeros((100,100), dtype='int')
    # seg[30:70, 30:70] = k
    # ground truth
    # gt = np.zeros((100,100), dtype='int')
    # gt[30:70, 40:80] = k
    if np.sum(gt) == 0:
        k = 0
        dice = np.sum(seg==gt) * 2.0 / (np.sum(seg==k) + np.sum(gt==k))
    else:
        num, list = count_diff_pixel_values(seg, 160, 160)
        print 'diff num', num,list
        dice = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))
    # print img_name,', dice similarity score: {}'.format(dice)
    return dice

