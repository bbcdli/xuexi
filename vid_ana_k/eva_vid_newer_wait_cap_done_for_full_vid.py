#eva_vid.py
# keras c3d eva_model
# !/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import c3d_keras_model as c3d_model
import sys
import keras.backend as K
import time
# import gizeh
##############################
# hy for saving video clips
from moviepy.editor import *
from moviepy.video.VideoClip import VideoClip
from moviepy.Clip import Clip
##############################
LOG_ON = False
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 2
CLIP_LENGTH = 16
TEST_VIDEO_LOAD_PATH = os.path.join(PROJ_DIR,'..','aggr_vids/')
EVA_SAVE_PATH_NO_AGGR = os.path.join(PROJ_DIR,'save_no_aggr/')
if not os.path.exists(EVA_SAVE_PATH_NO_AGGR):
 os.makedirs(EVA_SAVE_PATH_NO_AGGR)
EVA_SAVE_PATH = os.path.join(PROJ_DIR,'save_aggr/')
if not os.path.exists(EVA_SAVE_PATH):
 os.makedirs(EVA_SAVE_PATH)
v_dirs = sorted([s for s in os.listdir(TEST_VIDEO_LOAD_PATH) if '.mp4' in s
                 and ('02_Tweety im Zug105_115_F' in s)])
#01_fight_in_train_T
#04_Tweety im Zug_29_49_F
TEST_VIDEOS = []
for dir in v_dirs:
 v = TEST_VIDEO_LOAD_PATH + dir
 TEST_VIDEOS.append(v)
dim_ordering = K.image_dim_ordering()
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
 dim_ordering)
backend = dim_ordering
log_path = os.path.join(PROJ_DIR,'logs_hy/')
if not os.path.exists(log_path):
 os.makedirs(log_path)
str_log = ''

class Logger(object):
 def __init__(self, log_path, str_log):
  self.terminal = sys.stdout
  from datetime import datetime
  self.str_log = str_log
  self.log_path = log_path
  self.log = open(datetime.now().strftime(log_path + '%Y_%m_%d_%H_%M' + str_log + '.log'), "a")
 
 def write(self, message):
  self.terminal.write(message)
  self.log.write(message)
 
 def flush(self):
  # this flush method is needed for python 3 compatibility.
  # this handles the flush command by doing nothing.
  # you might want to specify some extra behavior here.
  pass

if LOG_ON:
 sys.stdout = Logger(log_path, str_log)

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
 if data.ndim > 2:
  if backend == 'th':
   data = np.transpose(data, (1, 2, 3, 0))
  # else:
  #    data = np.transpose(data, (0, 2, 1, 3))
  min_num_spatial_axes = 10
  max_outputs_to_show = 3
  ndim = data.ndim
  print "[Info] {}.ndim={}".format(label, ndim)
  print "[Info] {}.shape={}".format(label, data.shape)
  for d in range(ndim):
   num_this_dim = data.shape[d]
   if num_this_dim >= min_num_spatial_axes:  # check for spatial axes
    # just first, center, last indices
    range_this_dim = [0, num_this_dim / 2, num_this_dim - 1]
   else:
    # sweep all indices for non-spatial axes
    range_this_dim = range(num_this_dim)
   for i in range_this_dim:
    new_dim = tuple([d] + range(d) + range(d + 1, ndim))
    sliced = np.transpose(data, new_dim)[i, ...]
    print("[Info] {}, dim:{} {}-th slice: "
          "(min, max, mean, std)=({}, {}, {}, {})".format(
     label,
     d, i,
     np.min(sliced),
     np.max(sliced),
     np.mean(sliced),
     np.std(sliced)))
  if plots:
   # assume (l, h, w, c)-shaped input
   if data.ndim != 4:
    print("[Error] data (shape={}) is not 4-dim. Check data".format(
     data.shape))
    return
   l, h, w, c = data.shape
   if l >= min_num_spatial_axes or \
     h < min_num_spatial_axes or \
     w < min_num_spatial_axes:
    print("[Error] data (shape={}) does not look like in (l,h,w,c) "
          "format. Do reshape/transpose.".format(data.shape))
    return
   nrows = int(np.ceil(np.sqrt(data.shape[0])))
   # BGR
   if c == 3:
    for i in range(l):
     mng = plt.get_current_fig_manager()
     mng.resize(*mng.window.maxsize())
     plt.subplot(nrows, nrows, i + 1)  # doh, one-based!
     im = np.squeeze(data[i, ...]).astype(np.float32)
     im = im[:, :, ::-1]  # BGR to RGB
     # force it to range [0,1]
     im_min, im_max = im.min(), im.max()
     if im_max > im_min:
      im_std = (im - im_min) / (im_max - im_min)
     else:
      print "[Warning] image is constant!"
      im_std = np.zeros_like(im)
     plt.imshow(im_std)
     plt.axis('off')
     plt.title("{}: t={}".format(label, i))
    plt.show()
   # plt.waitforbuttonpress()
   else:
    for j in range(min(c, max_outputs_to_show)):
     for i in range(l):
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())
      plt.subplot(nrows, nrows, i + 1)  # doh, one-based!
      im = np.squeeze(data[i, ...]).astype(np.float32)
      im = im[:, :, j]
      # force it to range [0,1]
      im_min, im_max = im.min(), im.max()
      if im_max > im_min:
       im_std = (im - im_min) / (im_max - im_min)
      else:
       print "[Warning] image is constant!"
       im_std = np.zeros_like(im)
      plt.imshow(im_std)
      plt.axis('off')
      plt.title("{}: o={}, t={}".format(label, j, i))
     plt.show()
    # plt.waitforbuttonpress()
 elif data.ndim == 1:
  print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
   label,
   np.min(data),
   np.max(data),
   np.mean(data),
   np.std(data)))
  print("[Info] data[:10]={}".format(data[:10]))
 return

def main(model_name):
 show_images = False
 diagnose_plots = False
 count_correct = 0
 model_dir = os.path.join(PROJ_DIR,'log_models')
 global backend
 # override backend if provided as an input arg
 if len(sys.argv) > 1:
  if 'tf' in sys.argv[1].lower():
   backend = 'tf'
  else:
   backend = 'th'
 print "[Info] Using backend={}".format(backend)
 model_weight_filename = os.path.join(model_dir, model_name)
 model_json_filename = os.path.join(model_dir, 'sports1M_model_custom_2l.json')
 print("[Info] Reading model architecture...")
 model = model_from_json(open(model_json_filename, 'r').read())

 # visualize model
 model_img_filename = os.path.join(model_dir, 'c3d_model_custom.png')
 if not os.path.exists(model_img_filename):
  from keras.utils import plot_model
  plot_model(model, to_file=model_img_filename)
 model.load_weights(model_weight_filename)
 #print("[Info] Loading model weights -- DONE!")
 model.compile(loss='mean_squared_error', optimizer='sgd')
 #print("[Info] Loading labels...")
 with open('labels_aggr.txt', 'r') as f:
  labels_txt = [line.strip() for line in f.readlines()]
 print('Total labels: {}'.format(len(labels_txt)))
 
 for TEST_VIDEO in TEST_VIDEOS:
  print 'Test video path name:', TEST_VIDEO
  
  cap = cv2.VideoCapture(TEST_VIDEO)
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  print 'frame per second:', fps
  vid, vid_view, frame_i = [], [], 0
  while True:
   ret, img = cap.read()
   if ret:
    frame_i += 1
    if frame_i % 1 == 0:
     vid_view.append(img)
     vid.append(cv2.resize(img, (171, 128)))
   else:
    break
  total_video_frames = len(vid)
  print 'vid len', total_video_frames
  vid = np.array(vid, dtype=np.float32)
  # plt.imshow(vid[2000]/256)
  # plt.show()
  # sample 16-frame clip
  STOP = False
  total_test_clips = int(total_video_frames / CLIP_LENGTH)
  while not STOP:
   for num in xrange(total_test_clips):
    # start_frame = 2000
    start_frame = num * CLIP_LENGTH  # 0x16,1x16
    offset_time = start_frame / fps
    print '\nstart frame:{}, offset:{}'.format(start_frame,offset_time)
    X = vid[start_frame:(start_frame + CLIP_LENGTH), :, :, :]
    def eva_one_clip(X, start_frame, model, EVA_SAVE_PATH_NO_AGGR, EVA_SAVE_PATH,count_correct):
     # subtract mean
     do_sub_mean = False
     if do_sub_mean:
      mean_cube = np.load('models/train01_16_128_171_mean.npy')
      mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
      # diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
      X -= mean_cube
     # center crop
     X = X[:, 8:120, 30:142, :]  # (l, h, w, c)
     # diagnose(X, verbose=True, label='Center-cropped X', plots=show_images)
     if backend == 'th':
      X = np.transpose(X, (3, 0, 1, 2))  # input_shape = (3,16,112,112)
     else:
      pass  # input_shape = (16,112,112,3)
     # get activations for intermediate layers if needed
     inspect = False
     if inspect:
      inspect_layers = [
       #    'fc6',
       #    'fc7',
      ]
      for layer in inspect_layers:
       int_model = c3d_model.get_int_model(model=model, layer=layer, backend=backend)
       int_output = int_model.predict_on_batch(np.array([X]))
       int_output = int_output[0, ...]
       print "[Debug] at layer={}: output.shape={}".format(layer, int_output.shape)
       diagnose(int_output,
            verbose=True,
            label='{} activation'.format(layer),
            plots=diagnose_plots,
            backend=backend)
     # inference
     start_p = time.time()
     output = model.predict_on_batch(np.array([X]))
     end_p = time.time()
     print 'time elapsed for model.predict:{:.5} s'.format(end_p - start_p)
     max_output = max(output[0])
     # if max_output < 0.5:
     #   EVA_SAVE_PATH = EVA_SAVE_PATH_NO_AGGR
     # EVA_SAVE_PATH = EVA_SAVE_PATH_NO_AGGR #setting save type
     v_str = os.path.splitext(os.path.basename(TEST_VIDEO))[0]
     clip_name = EVA_SAVE_PATH + v_str + '_' + str(start_frame) + '_' + "%.3f" % max_output + '.mp4'
     filename_f = EVA_SAVE_PATH + v_str + '_' + str(start_frame) + '_' + "%.3f" % max_output + '.jpg'
     # pred_label = output[0].argmax()
     indx_of_interest = start_frame
     print 'index of interest:', indx_of_interest
     
     def save_current_subclips_to_frames(v_str):
      for frame, i in zip(vid_view[start_frame:start_frame + 16], xrange(CLIP_LENGTH)):
       filename_f_i = EVA_SAVE_PATH + 'eva_' + v_str + '_' + str(start_frame) + '_' + "%.3f" % max_output + str(
        i) + '.png'
       cv2.imwrite(filename_f_i, frame)
      # if max_output > 0.4:
      #   save_current_subclips_to_frames()
     
     def save_start_frame_of_interest(vid_view, indx_of_interest, filename_f):
      frame_save = vid_view[indx_of_interest]
      # cv2.imshow(filename_f,frame_save)
      cv2.imwrite(filename_f, frame_save)
     
     def save_subclip(TEST_VIDEO, indx_of_interest, fps):
      clip = VideoFileClip(TEST_VIDEO)
      v_time = indx_of_interest / fps
      print 'Position of maximum probability:{}'.format(indx_of_interest)
      print 'aggr high time point: {}'.format(v_time)
      subclip = clip.subclip(v_time - 8, v_time + 2)  # 74.6-8, 76 set an interval around frame of interest
      subclip.write_videofile(clip_name)
      cv2.waitKey(130)
     
     if max_output > 0.3 or max_output < 0.1:
      # if max_output > 0.2 or max_output < 0.08: #for no aggr
      # save_subclip(TEST_VIDEO,indx_of_interest,fps)
      # save_start_frame_of_interest(vid_view,indx_of_interest,filename_f)
      #save_current_subclips_to_frames(v_str)
       pass
     # show results
     save_probability_to_png = False
     if save_probability_to_png:
      print('Saving class probabilities in probabilities.png')
      plt.plot(output[0])
      plt.title('Probability')
      plt.savefig('probabilities_'+v_str+'.png')
      print('Maximum probability: {:.5f}'.format(max(output[0])))
      print('Predicted label: {}'.format(labels_txt[output[0].argmax()]))
     # sort top five predictions from softmax output
     top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
     top_inds_list = top_inds.tolist()
     print '\nTop probabilities and labels:',top_inds #array([0, 1])
     print 'out[0][0]:{:0.4f}'.format(output[0][0])
     print 'out[0][1]:{:0.4f}'.format(output[0][1])
     '''
     for i,index in zip(top_inds,xrange(NUM_CLASSES)): #[1,0] [0,1]
      #print('{1}: {0:.5f}  other'.format(int(output[0][i]), labels_txt[i]))
      if index == 0:
       if i == gt_label:
        count_correct += 1
        print labels_txt[i],': {:0.4f}'.format(output[0][i]),'  top1  correct ',count_correct
       else:
        print labels_txt[i],': {:0.4f}'.format(output[0][i]),'  top1'
      else:
       #print('{1}: {0:.5f}  other'.format(int(output[0][i]), labels_txt[i]))
       print labels_txt[i], ': {:0.4f}'.format(output[0][i]), '  other'
     '''
     return count_correct
    count_correct = eva_one_clip(X, start_frame, model, EVA_SAVE_PATH_NO_AGGR, EVA_SAVE_PATH,count_correct)
    print 'Precision:{:.3f}'.format(count_correct/float(total_test_clips))
    if (total_video_frames - start_frame) < CLIP_LENGTH * 2:
     print 'set STOP-True'
     STOP = True
     break
 print 'TEST end.'

if __name__ == '__main__':
  #model_name = 'k_01-0.46.hdf5'#offi
  model_name = 'No1_k_01-0.62_0.27.hdf5'
  main(model_name)

