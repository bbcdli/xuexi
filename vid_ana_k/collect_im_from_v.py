import cv2
import os,sys,time
from random import randint
import numpy as np
import c3d_keras_model as c3d_model

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
V_READ_PATH = '/home/hy/Documents/hy_dev/aggr/aggr_vids/'
TMP_SAVE_PATH = os.path.join(PROJ_DIR, 'tmp')
if not os.path.exists(TMP_SAVE_PATH):
  os.makedirs(TMP_SAVE_PATH)

SAVE_PATH = '/home/hy/hy_dev/aggr/Own_data'
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)

T_dir = sorted([s for s in os.listdir(V_READ_PATH) if '_T.mp4' in s])
F_dir = sorted([s for s in os.listdir(V_READ_PATH) if '_F.mp4' in s])

clip_len = 16

def reduce_mean_stdev(images, print_val=False):
  mean = np.mean(images)
  stdev = np.std(images)
  if print_val:
    print 'mean %d,stdev %d', (mean, stdev)
  images = images - mean
  images_reduced_mean = images / stdev
  return images_reduced_mean

def dense_to_one_hot(labels_dense, num_classes=0):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]  # num_labels is the same as num of images
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  # each label is written as one vector:eg. class 0 of total 6 classes is [1,0,0,0,0,0]
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def collect_1_subclip_to_imgs(vid,start_frame,crop_size,V_FILE):
  clipdata, img_datas, label_clips, clipsdata = [], [], [], []
  for img, i in zip(vid[start_frame:start_frame + 16], xrange(clip_len)):
    crop_x = int((img.shape[0] - crop_size) / 2)
    crop_y = int((img.shape[1] - crop_size) / 2)
    img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
    img = reduce_mean_stdev(img)
    img_datas.append(img)

  tmp_label = 1 if '_T.' in V_FILE else 0
  return img_datas, tmp_label  # one pair:clip+label

def collect_subclip_to_frames_rnd(vid,num_of_clips_to_gen,fps,V_FILE):
  clipsdata, label_clips, start_frames, offset_times, crop_size = [], [], [],[], 112

  for num in xrange(num_of_clips_to_gen):
    rnd_frame_start = randint(0, len(vid) - clip_len - 1)
    start_frames.append(rnd_frame_start)  # 0x16,1x16
    offset_time = rnd_frame_start / fps
    offset_times.append(offset_time)
    #print '\nclip: {},  start frame:{}'.format(num,start_frame)
    #print 'offset:', offset_time
    # X = vid_np[start_frame:(start_frame + clip_len), :, :, :]

    images_data, tmp_label = collect_1_subclip_to_imgs(vid, rnd_frame_start, crop_size,V_FILE)
    clipsdata.append(images_data)
    label_clips.append(int(tmp_label))

  np_arr_data = np.array(clipsdata).astype(np.float32)
  np_arr_label = np.array(label_clips).astype(np.int64)
  np_arr_label_onehot = dense_to_one_hot(np_arr_label, c3d_model.NUM_CLASSES)
  print 'start frames:{}'.format(start_frames),'\n','offset_times:{:.6}'.format(offset_times)
  return np_arr_data, np_arr_label_onehot

def collect_train_data(num_of_clips_pro_class):
  #T_dir[0] = '03_Dog_lover_knocks_out_a_dog_abuser637_T.mp4'
  #F_dir[0] = '01_Classroom_management-Week10224_0244_F.mp4'
  T_dir[0] = '04_Tschetschenischer_Tuersteher_185_202_T.mp4' #01_TschClubMBarGo_130_144_T
  F_dir[0] = '01_Classroom_management-Week10224_0244_F.mp4'
  images_np, labels_np = collect_class_data(num_of_clips_pro_class,V_FILE = T_dir[0])
  #images_np2, labels_np2 = collect_class_data(num_of_clips_pro_class/2,V_FILE = T_dir[1])

  #images_np = np.concatenate((images_np, images_np2), axis=0)
  #labels_np = np.concatenate((labels_np, labels_np2), axis=0)

  images_np3, labels_np3 = collect_class_data(num_of_clips_pro_class,V_FILE = F_dir[0])

  images_np = np.concatenate((images_np,images_np3),axis=0)
  labels_np = np.concatenate((labels_np,labels_np3),axis=0)
  return images_np, labels_np

def collect_test_data(num_of_clips_pro_class):
  test_pkg = 'A'
  if test_pkg == 'A':
    T_dir[1] = '04_Tschetschenischer_Tuersteher_140_147_T.mp4'
    F_dir[1] = '01_UBahn-S_650_665_F.mp4'
  if test_pkg == 'B':
    T_dir[1] = '01_TschClubMBarGo_30_38_T.mp4'
    F_dir[1] = '01_Classroom_management-Week10224_0244_F.mp4'

  images_np, labels_np = collect_class_data(num_of_clips_pro_class,V_FILE = T_dir[1])
  images_np1, labels_np1 = collect_class_data(num_of_clips_pro_class,V_FILE = F_dir[1])
  images_np = np.concatenate((images_np,images_np1),axis=0)
  labels_np = np.concatenate((labels_np,labels_np1),axis=0)

  return images_np, labels_np

def collect_class_data(num_of_clips_to_gen,V_FILE):
  print 'V_FILE:',V_FILE
  cap = cv2.VideoCapture(os.path.join(V_READ_PATH,V_FILE))
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

  print 'frame per second:', fps
  vid, vid_view, frame_i = [], [], 0
  while True:
    ret, img = cap.read()
    if ret:
      frame_i += 1
      #cv2.imshow('frame',img)
      #cv2.waitKey(5)
      if frame_i % 1 == 0:
        vid_view.append(img)
        vid.append(cv2.resize(img, (171, 128)))
    else:
      break

  total_video_frames = len(vid)
  print 'vid len', total_video_frames
  vid_np = np.array(vid, dtype=np.float32)
  STOP = False
  while not STOP:
    np_arr_data, np_arr_label_onehot = \
        collect_subclip_to_frames_rnd(vid,num_of_clips_to_gen,fps,V_FILE)
    print 'np_arr_data shape:',np_arr_data.shape
    STOP = True

  return np_arr_data, np_arr_label_onehot
