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
global LOG_ON,PROJ_DIR,NUM_CLASSES,CLIP_LENGTH,TEST_VIDEO_LOAD_PATH,TEST_VIDEOS
global EVA_SAVE_PATH,EVA_SAVE_PATH_NO_AGGR,SAVE_RES_IMAGES
LOG_ON = False
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 2
CLIP_LENGTH = 16
SAVE_RES_IMAGES = True
#os.path.dirname(os.path.abspath(__file__))
# TEST_V_PATH = PROJ_DIR + 'test_videos/'
#os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'templates'))
TEST_VIDEO_LOAD_PATH = os.path.join(PROJ_DIR,'..','aggr_vids/')#'/home/hy/Documents/aggr/aggr_vids/'
#TEST_VIDEO_LOAD_PATH = '/media/sf_shared_win/vids/'
EVA_SAVE_PATH_NO_AGGR = os.path.join(PROJ_DIR,'save_no_aggr/')
if not os.path.exists(EVA_SAVE_PATH_NO_AGGR):
 os.makedirs(EVA_SAVE_PATH_NO_AGGR)
EVA_SAVE_PATH = os.path.join(PROJ_DIR,'save_aggr/')
if not os.path.exists(EVA_SAVE_PATH):
 os.makedirs(EVA_SAVE_PATH)
# v_dirs = sorted([s for s in os.listdir(TEST_V_PATH) if '2017-09-06_13.57.41.3.cam_55_3.event50T.mp4' in s ])
v_dirs = sorted([s for s in os.listdir(TEST_VIDEO_LOAD_PATH) if '.' in s
                 and ('02_Tschetschene_vs_Russe_T' in s)])
TEST_VIDEOS = []
for dir in v_dirs:
 v = TEST_VIDEO_LOAD_PATH + dir
 TEST_VIDEOS.append(v)
# TEST_VIDEOS = TEST_VIDEOS[0:2]
# TEST_VIDEO = PROJ_DIR+'test_videos/2017-09-06_event57F.mp4'
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

#display
def demo(frame,pred_txt,pred_label, confidence):#pred_txt,pred_label, gt_label
 if pred_txt == 'aggr':
  # print 'RES:', RES, 'target', target
  tp_color = (0, 0, 255)
 elif pred_txt == 'normal':
  tp_color = (255,255,0)
 else:
  tp_color = (0, 255, 255)
 prob_str = str(confidence)#'todo'
 cv2.putText(frame, ">:" + pred_txt,
             org=(int(frame.shape[1]*0.1), int(frame.shape[0] * 0.15)),
             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=tp_color, thickness=2)
 cv2.putText(frame, ">:" + prob_str,
             org=(int(frame.shape[1]*0.1), int(frame.shape[0] * 0.3)),
             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=tp_color, thickness=2)
 cv2.imshow('demo',frame)
 cv2.waitKey(5)


def eva_one_clip(X, vid_view,start_frame,model, EVA_SAVE_PATH_NO_AGGR,
                 EVA_SAVE_PATH, count_correct,labels_txt,gt_label,TEST_VIDEO):
 # subtract mean
 do_sub_mean = False
 if do_sub_mean:
  mean_cube = np.load('models/train01_16_128_171_mean.npy')
  mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
  # diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
  X -= mean_cube
 # center crop
 #X = X[:, 8:120, 30:142, :]  # (l, h, w, c)
 # diagnose(X, verbose=True, label='Center-cropped X', plots=show_images)
 if backend == 'th':
  X = np.transpose(X, (3, 0, 1, 2))  # input_shape = (3,16,112,112)
 else:
  pass  # input_shape = (16,112,112,3)
 # get activations for intermediate layers if needed
 inspect = False
 if inspect:
  inspect_layers = [ #    'fc6',   #    'fc7',
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
 v_str = EVA_SAVE_PATH + v_str
 # pred_label = output[0].argmax()
 indx_of_interest = start_frame
 print 'index of interest:', indx_of_interest


 def save_current_subclips_to_frames(vid_view,v_str,start_frame):
  for frame, i in zip(vid_view[0:CLIP_LENGTH],
                      xrange(start_frame,start_frame + CLIP_LENGTH)):
   filename_f_i = v_str +'fr_' + '%04d'%(i) + '.png'
   #print filename_f_i
   cv2.imwrite(filename_f_i, frame)


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
  cv2.waitKey(10)


 if max_output > 0.3 or max_output < 0.1:
  # if max_output > 0.2 or max_output < 0.08: #for no aggr
  # save_subclip(TEST_VIDEO,indx_of_interest,fps)
  # save_start_frame_of_interest(vid_view,indx_of_interest,filename_f)
  # save_current_subclips_to_frames(v_str)
  pass
 # show results
 save_probability_to_png = False
 if save_probability_to_png:
  print('Saving class probabilities in probabilities.png')
  plt.plot(output[0])
  plt.title('Probability')
  plt.savefig('probabilities_' + v_str + '.png')
  print('Maximum probability: {:.5f}'.format(max(output[0])))
  print('Predicted label: {}'.format(labels_txt[output[0].argmax()]))

 # sort top five predictions from softmax output
 top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
 top_inds_list = top_inds.tolist()

 print '\nTop probabilities and labels:', top_inds  # array([0, 1])
 pred_txt = 'aggr' if top_inds[0] == 1 else 'normal'
 p = max(output[0])
 print('p: {:.5f}'.format(max(output[0])))

 # print 'out[0][0]:{:0.4f}'.format(output[0][0])
 # print 'out[0][1]:{:0.4f}'.format(output[0][1])

 for i, index in zip(top_inds, xrange(NUM_CLASSES)):  # [1,0] [0,1]
  # print('{1}: {0:.5f}  other'.format(int(output[0][i]), labels_txt[i]))
  if index == 0:
   if i == gt_label:
    count_correct += 1
    print labels_txt[i], ': {:0.4f}'.format(output[0][i]), '  top1  correct ', count_correct
   else:
    print labels_txt[i], ': {:0.4f}'.format(output[0][i]), '  top1 wrong'
    if SAVE_RES_IMAGES and p>0.99:
     print 'save frames'
     save_current_subclips_to_frames(vid_view,v_str,start_frame)
  else:
   # print('{1}: {0:.5f}  other'.format(int(output[0][i]), labels_txt[i]))
   print labels_txt[i], ': {:0.4f}'.format(output[0][i]), '  rank lower'
 return count_correct,pred_txt,top_inds[0],p

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
 #if backend == 'th':
 model_weight_filename = os.path.join(model_dir, model_name)
 # model_json_filename = os.path.join(model_dir, 'sports1M_model_custom_3l.json')
 print("[Info] Reading model architecture...")
 #model = model_from_json(open(model_json_filename, 'r').read())
 from keras.models import load_model
 model = load_model(model_weight_filename)
 # model = c3d_model.get_model(backend=backend)
 # visualize model
 model_img_filename = os.path.join(model_dir, 'c3d_model_custom.png')
 if not os.path.exists(model_img_filename):
  from keras.utils import plot_model
  plot_model(model, to_file=model_img_filename)
 #print("[Info] Loading model weights...")
 model.load_weights(model_weight_filename)
 #print("[Info] Loading model weights -- DONE!")
 model.compile(loss='mean_squared_error', optimizer='sgd')
 #print("[Info] Loading labels...")
 with open('labels_aggr.txt', 'r') as f:
  labels_txt = [line.strip() for line in f.readlines()]
 print('Total num of classes: {}'.format(len(labels_txt)))
 
 gt_label = 0
 cap = cv2.VideoCapture(0)
 cap.set(1,2)
 fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
 print 'frame per second:', fps
 vid, vid_view, frame_i = [], [], 0
 STOP,clip = False,[]
 while True:
  ret, img = cap.read()
  k = cv2.waitKey(30) & 0xFF
  if k == ord('q'):
   print 'key interrupted'
   break
   exit(0)
  if ret:
   frame_i += 1
   if frame_i % 1 == 0:
    vid_view.append(img)
    #vid.append(cv2.resize(img, (171, 128)))
    vid.append(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))
    if frame_i % CLIP_LENGTH == 0 and frame_i> CLIP_LENGTH:
     print 'frame:',frame_i
     vid_np = np.array(vid, dtype=np.float32)
     start_frame = frame_i - CLIP_LENGTH
     X = vid_np[0:(CLIP_LENGTH), :, :, :]
     offset_time = start_frame / fps
     print '\nstart frame:{}, offset:{}'.format(start_frame,offset_time)

     total_video_frames = frame_i
     total_test_clips = int(total_video_frames / CLIP_LENGTH)
     print 'vid len', total_video_frames
     # plt.show()
     count_correct,pred_txt,pred_label,confidence = eva_one_clip(X, vid_view,start_frame, model,
                     EVA_SAVE_PATH_NO_AGGR, EVA_SAVE_PATH,count_correct,
                     labels_txt,gt_label,TEST_VIDEO)
     print 'Precision:{:.3f}'.format(count_correct/float(total_test_clips))
     #for i in xrange(CLIP_LENGTH):
     #demo(vid_view[i], pred_txt,pred_label, confidence)
     demo(vid_view[(frame_i%CLIP_LENGTH)], pred_txt,pred_label, confidence)
     vid,vid_view = [],[]

  else:
   cap.release()
   break
 print 'TEST end.'

if __name__ == '__main__':
  #model_name = 'k_01-0.46.hdf5'#offi
  global INPUT_SIZE,gt_label_webcam,TEST_VIDEO
  INPUT_SIZE = 112
  gt_label_webcam = '_'
  TEST_VIDEO = ''
  model_name = 'k_16_00000314_03-0.51best.hdf5'
  main(model_name,)

#https://www.youtube.com/watch?v=_q02y7UiFc0 #Tschetschenen stuermen Russischen Club "MBarGo"