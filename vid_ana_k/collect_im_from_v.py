import cv2
import os,sys,time
from random import randint
import numpy as np

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
#V_READ_PATH = '/home/hy/Documents/hy_dev/aggr/aggr_vids/'
V_READ_PATH = os.path.join(PROJ_DIR,'..', 'aggr_vids/')
TMP_SAVE_PATH = os.path.join(PROJ_DIR, 'tmp')
if not os.path.exists(TMP_SAVE_PATH):
  os.makedirs(TMP_SAVE_PATH)

SAVE_PATH = '/home/hy/hy_dev/aggr/Own_data'
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)

T_dir = sorted([s for s in os.listdir(V_READ_PATH) if '_T.mp4' in s])
F_dir = sorted([s for s in os.listdir(V_READ_PATH) if '_F.mp4' in s])


# cv2 resize: downsampling.interpolation=cv2.INTER_AREA upsampling interpolation=cv2.INTER_CUBIC
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

def collect_1_subclip_to_imgs(vid,clip_len,start_frame,crop_size,V_FILE,FLIP):
  clipdata, img_datas, label_clips, clipsdata = [], [], [], []
  for img, i in zip(vid[start_frame:start_frame + 16], xrange(clip_len)):
    #crop_x = int((img.shape[0] - crop_size) / 2)
    #crop_y = int((img.shape[1] - crop_size) / 2)
    #img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]

    #img = cv2.resize(img,(crop_size,crop_size),interpolation=cv2.INTER_AREA)
    #augmentation
    #flip
    #FLIP = False
    img_copy = img.copy()
    if FLIP:
      for fl in xrange(-1,2):
        img_f = cv2.flip(img_copy,fl)
        img_datas.append(img_f)
    img_datas.append(img)
    img_datas_reduce_mean = reduce_mean_stdev(img_datas)
    #img = reduce_mean_stdev(img)

  #tmp_label = 1 if '_T.' in V_FILE else 0
  return img_datas_reduce_mean #img_datas  # for one pair or 4 pairs with augemnted:clip+label

def collect_subclip_to_frames_rnd(vid,
                                  clip_len,
                                  input_size,num_classes,
                                  num_of_clips_to_gen,fps,V_FILE,CLASS,FLIP):
  clipsdata, label_clips, start_frames, offset_times, crop_size = [], [], [],[], input_size
  datas= []
  np_arr_label_onehot = np.array([], dtype=np.int64).reshape(0, num_classes)

  for num in xrange(num_of_clips_to_gen):
    rnd_frame_start = randint(0, len(vid) - clip_len - 1)
    start_frames.append(rnd_frame_start)  # 0x16,1x16
    offset_time = rnd_frame_start / fps
    offset_times.append(offset_time)
    #print '\nclip: {},  start frame:{}'.format(num,start_frame)
    #print 'offset:', offset_time
    # X = vid_np[start_frame:(start_frame + clip_len), :, :, :]

    images_data = collect_1_subclip_to_imgs(vid,clip_len,rnd_frame_start,crop_size,V_FILE,FLIP)
    #images_data contains 4*num_clips*16 frames
    clipsdata.append(images_data)
    n = len(images_data)/clip_len
    label_clips.append(int(CLASS))
    for i in xrange(n):
      datas.append(np.array(images_data[i*clip_len:(i+1)*clip_len]).astype(np.float32))

  for i in xrange(n):
    np_arr_label = (np.array(label_clips).astype(np.int64))
    #label_onehot = dense_to_one_hot(np_arr_label, num_classes)
    np_arr_label_onehot = np.concatenate((np_arr_label_onehot,dense_to_one_hot(np_arr_label, num_classes)),axis=0)
    #print '\nnp.asarray(label_onehots):', (label_onehots).shape

  print 'start frames:{}'.format(start_frames)
  #print 'offset_times:{:.6}'.format(offset_times)
  return np.asarray(datas), np_arr_label_onehot
  #return np_arr_data, np_arr_label_onehot


def collect_subclip_to_frames_rnd_BACKUP(vid,
                                  clip_len,
                                  input_size,num_classes,
                                  num_of_clips_to_gen,fps,V_FILE,CLASS,FLIP):
  clipsdata, label_clips, start_frames, offset_times, crop_size = [], [], [],[], input_size

  for num in xrange(num_of_clips_to_gen):
    rnd_frame_start = randint(0, len(vid) - clip_len - 1)
    start_frames.append(rnd_frame_start)  # 0x16,1x16
    offset_time = rnd_frame_start / fps
    offset_times.append(offset_time)
    #print '\nclip: {},  start frame:{}'.format(num,start_frame)
    #print 'offset:', offset_time
    # X = vid_np[start_frame:(start_frame + clip_len), :, :, :]

    images_data = collect_1_subclip_to_imgs(vid,clip_len,rnd_frame_start,crop_size,V_FILE,FLIP)
    clipsdata.append(images_data)
    label_clips.append(int(CLASS))

  np_arr_data = np.array(clipsdata).astype(np.float32)
  np_arr_label = np.array(label_clips).astype(np.int64)
  np_arr_label_onehot = dense_to_one_hot(np_arr_label, num_classes)
  print 'start frames:{}'.format(start_frames)
  #print 'onehot:',np_arr_label_onehot.shape #(10,num_classes)
  #print 'offset_times:{:.6}'.format(offset_times)
  return np_arr_data, np_arr_label_onehot


def collect_class_data(clip_len,input_size,num_classes,num_of_clips_to_gen,V_FILE,CLASS,FLIP):
  print 'V_FILE:',V_FILE
  cap = cv2.VideoCapture(os.path.join(V_READ_PATH,V_FILE))
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

  #print 'frame per second:', fps
  vid, vid_view, frame_i = [], [], 0
  while True:
    ret, img = cap.read()
    if ret:
      frame_i += 1
      if frame_i % 1 == 0:
        vid_view.append(img)
        #vid.append(cv2.resize(img, (171, 128)))
        vid.append(cv2.resize(img, (input_size, input_size)))
    else:
      break

  total_video_frames = len(vid)
  #vid_np = np.array(vid, dtype=np.float32)
  STOP = False
  while not STOP:
    np_arr_data, np_arr_label_onehot = \
        collect_subclip_to_frames_rnd(vid,clip_len,input_size,
                                      num_classes,num_of_clips_to_gen,fps,
                                      V_FILE,CLASS,FLIP)
    print 'np_data shape:',np_arr_data.shape, ', vid len:',total_video_frames
    STOP = True
  cap.release()

  return np_arr_data, np_arr_label_onehot

def collect_train_data(clip_len,input_size,num_classes,num_of_clips_pro_class):
  T_dir[0] = 'LatinKingsvsTangoBlastHoustone_T.mp4'#'03_Dog_lover_knocks_out_a_dog_abuser637_T.mp4'
  F_dir[0] = '01_Classroom_management_Week10224_0244_F.mp4' #'04_879_38_l_good_F.mov' #classroom(480)

  T_dir[1] = '04_Tschetschenischer_Tuersteher_185_202_T.mp4' #01_TschClubMBarGo_130_144_T
  F_dir[1] = '01_UBahn_S_485_498_F.mp4'

  T_dir[2] = '01_TschClubMBarGo_130_144_T.mp4'  # 01_TschClubMBarGo_130_144_T
  F_dir[2] = '04_Tweety_im_Zug_29_49_F.mp4'

  T_dir[3] = '01_TschClubMBarGo_415_444_T.mp4'
  F_dir[3] = 'priv_F.mp4'

  T_dir[4] = '01_fight_in_train_70_99_T.mp4'
  F_dir[4] = '01_UBahn_S_650_665_F.mp4'

  #
  #TschClubMBarGo_30_38_T
  # 600,480,  425,325,  350,400,  725,480,  869,375
  num_of_clips1, num_of_clips2, num_of_clips3,num_of_clips4,num_of_clips5\
    = 10, 5,5,18,12#for 220: 74, 24, 18, 64, 40 #40, 24, 8,40
  total_vids = 5
  images_np, labels_np = collect_class_data(clip_len,input_size,num_classes,num_of_clips1,
    V_FILE = T_dir[0],CLASS=1,FLIP=True)

  images_np_f1, labels_np_f1 = collect_class_data(clip_len,input_size,num_classes,int(num_of_clips_pro_class/total_vids),
    V_FILE = F_dir[0],CLASS=0,FLIP=True)

  images_np = np.concatenate((images_np, images_np_f1), axis=0)
  labels_np = np.concatenate((labels_np, labels_np_f1), axis=0)
  #########################################################
  images_np2, labels_np2 = collect_class_data(clip_len,input_size,num_classes,num_of_clips2,
    V_FILE = T_dir[1],CLASS=1,FLIP=True)

  images_np_f2, labels_np_f2 = collect_class_data(clip_len,input_size,num_classes,int(num_of_clips_pro_class/total_vids),
    V_FILE = F_dir[1],CLASS=0,FLIP=True)

  images_np = np.concatenate((images_np, images_np2), axis=0)
  labels_np = np.concatenate((labels_np, labels_np2), axis=0)

  images_np = np.concatenate((images_np,images_np_f2), axis=0)
  labels_np = np.concatenate((labels_np,labels_np_f2), axis=0)
  #########################################################

  images_np3, labels_np3 = collect_class_data(clip_len,input_size,num_classes,
              num_of_clips3,
              V_FILE=T_dir[2], CLASS=1,FLIP=True)

  images_np_f3, labels_np_f3 = collect_class_data(clip_len,input_size,num_classes,
              int(num_of_clips_pro_class /total_vids),
              V_FILE=F_dir[2], CLASS=0,FLIP=True)

  images_np = np.concatenate((images_np, images_np3), axis=0)
  labels_np = np.concatenate((labels_np, labels_np3), axis=0)

  images_np = np.concatenate((images_np, images_np_f3), axis=0)
  labels_np = np.concatenate((labels_np, labels_np_f3), axis=0)
  #########################################################
  images_np4, labels_np4 = collect_class_data(clip_len,input_size,num_classes,
              num_of_clips4,
              V_FILE=T_dir[3], CLASS=1,FLIP=True)

  images_np_f4, labels_np_f4 = collect_class_data(clip_len,input_size,num_classes,
              int(num_of_clips_pro_class/total_vids),
              V_FILE=F_dir[3], CLASS=0,FLIP=True)

  images_np = np.concatenate((images_np, images_np4), axis=0)
  labels_np = np.concatenate((labels_np, labels_np4), axis=0)


  images_np = np.concatenate((images_np, images_np_f4), axis=0)
  labels_np = np.concatenate((labels_np, labels_np_f4), axis=0)
  #########################################################

  images_np5, labels_np5 = collect_class_data(clip_len,input_size,num_classes,
              num_of_clips5,
              V_FILE=T_dir[4], CLASS=1,FLIP=True)

  images_np_f5, labels_np_f5 = collect_class_data(clip_len,input_size,num_classes,
              int(num_of_clips_pro_class / total_vids),
              V_FILE=F_dir[4], CLASS=0,FLIP=True)

  images_np = np.concatenate((images_np, images_np5), axis=0)
  labels_np = np.concatenate((labels_np, labels_np5), axis=0)

  images_np = np.concatenate((images_np, images_np_f5), axis=0)
  labels_np = np.concatenate((labels_np, labels_np_f5), axis=0)
  #########################################################

  return images_np, labels_np

def collect_test_data(clip_len,input_size,num_classes,num_of_clips_pro_class):
  #175,400,  200,240
  T_dir = '04_Tschetschenischer_Tuersteher_140_147_T.mp4'
  F_dir = 'priv_F.mp4'

  images_np, labels_np = collect_class_data(clip_len,input_size,num_classes,num_of_clips_pro_class,
                                            V_FILE = T_dir,CLASS=1,FLIP=False)
  images_np_f1, labels_np_f1 = collect_class_data(clip_len,input_size,num_classes,num_of_clips_pro_class,
                                              V_FILE = F_dir,CLASS=0,FLIP=False)

  images_np = np.concatenate((images_np,images_np_f1),axis=0)
  labels_np = np.concatenate((labels_np,labels_np_f1),axis=0)

  T_dir = 'TschClubMBarGo_30_38_T.mp4'
  F_dir = '02_Tweety_im_Zug105_115_F.mp4'

  images_np2, labels_np2 = collect_class_data(clip_len,input_size,num_classes,num_of_clips_pro_class,
                                            V_FILE=T_dir, CLASS=1,FLIP=False)
  images_np_f2, labels_np_f2 = collect_class_data(clip_len,input_size,num_classes,num_of_clips_pro_class,
                                                  V_FILE=F_dir, CLASS=0,FLIP=False)

  images_np = np.concatenate((images_np, images_np2), axis=0)
  labels_np = np.concatenate((labels_np, labels_np2), axis=0)

  images_np = np.concatenate((images_np, images_np_f2), axis=0)
  labels_np = np.concatenate((labels_np, labels_np_f2), axis=0)

  return images_np, labels_np

