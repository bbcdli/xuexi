#train_k.py
from keras.models import model_from_json
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.optimizers import SGD
import c3d_keras_model_newer as c3d_model
import numpy as np
import os, sys, time, random
import cv2
from PIL import Image
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def rot90(W):
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      for k in range(W.shape[2]):
        W[i, j, k] = np.rot90(W[i, j, k], 2)
  return W

# collect images of given path for one clip
def get_frames_data_read_clipline(lkdir, num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  # print 'lkdir', lkdir
  im_dirs = sorted([s for s in os.listdir(lkdir)])
  s_index = random.randint(0, len(im_dirs) - num_frames_per_clip)
  if len(im_dirs) < num_frames_per_clip:
    return [], s_index
  for i, im in zip(range(s_index, s_index + num_frames_per_clip), im_dirs):
    image_name = lkdir + '/' + im
    # print 'image:', image_name
    img = Image.open(image_name)
    img_data = np.array(img)
    ret_arr.append(img_data)
  return ret_arr, s_index

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

def read_clip_and_label_v2(filename, read_size, start_pos=-1, num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP, crop_size=112,
                           shuffle=False):
  lines = open(filename, 'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  # print 'len lines', len(lines)
  # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if (batch_index >= read_size):
      next_batch_start = index
      # print 'batch pos > batch_size', batch_index
      break
    # split line by ' '
    line = lines[index].strip('\n').split()
    dir = line[0]
    tmp_label = line[1]
    if not shuffle:
      # print("Loading a video clip from {}...".format(dir))
      pass  # hy
    tmp_data, _ = get_frames_data_read_clipline(dir, num_frames_per_clip)
    # print 'got frames data, len data', len(tmp_data)
    img_datas = []
    if (len(tmp_data) != 0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if (img.width > img.height):
          scale = float(crop_size) / float(img.height)
          img = np.array(cv2.resize(np.array(img), (112, 112))).astype(np.float32)
        else:
          scale = float(crop_size) / float(img.width)
          img = np.array(cv2.resize(np.array(img), (112, 112))).astype(np.float32)
          # img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        np_mean = reduce_mean_stdev(img)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean
        img_datas.append(img)
      data.append(img_datas)
      # tmp_label_list = list(int(tmp_label))
      np_arr_data = np.array(data).astype(np.float32)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dir)
  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = read_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      # convert to onehot
      label.append(int(tmp_label))
      np_arr_label = np.array(label).astype(np.int64)
      np_arr_label_onehot = dense_to_one_hot(np_arr_label, c3d_model.NUM_CLASSES)
  # np_arr_data = np.array(data).astype(np.float32)
  # np_arr_label = np.array(label).astype(np.int64)
  # convert to onehot
  # print 'len of label:',len(label),label
  np_arr_label = np.array(label).astype(np.int64)
  np_arr_label_onehot = dense_to_one_hot(np_arr_label, c3d_model.NUM_CLASSES)
  # print 'label_onehot:',np_arr_label_onehot
  # print 'next' ,next_batch_start
  return np_arr_data, np_arr_label_onehot, next_batch_start, read_dirnames, valid_len

def train(MAX_ITERATION, learning_rate, dropouts,model_path, model_name,log_LABEL, batch_size,
          momentum,num_of_clips_pro_class):
  # if 'con' in train_mode:
  dim_ordering = K.image_dim_ordering()
  print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
    dim_ordering)
  backend = dim_ordering
  #import caffe_pb2 as caffe
  #import numpy as np
  #p = caffe.NetParameter()
  #p.ParseFromString(
  #  open('models/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read()
  #)
  '''
  params = []
  conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
  fc_layers_indx = [22, 25, 28]
  for i in conv_layers_indx:
      layer = p.layers[i]
      weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
      weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
          layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
          layer.blobs[0].height, layer.blobs[0].width
      )
      weights_p = rot90(weights_p)
      params.append([weights_p, weights_b])
  for i in fc_layers_indx:
      layer = p.layers[i]
      weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
      weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
          layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
          layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T
      params.append([weights_p, weights_b])
  model_layers_indx = [0, 2, 4, 5, 7, 8, 10, 11] + [15, 17, 19] #conv + fc
  for i, j in zip(model_layers_indx, range(11)):
      model.layers[i].set_weights(params[j])
  '''
  keras.callbacks.History()
  epochs = MAX_ITERATION  #
  # decay_rate = learning_rate / epochs #not so good
  read_from_im_folder = False
  if read_from_im_folder:
    read_size = 56# 56 num of total clips if read images
    train_images, train_labels, next_tr, _, _ = read_clip_and_label_v2(
      filename=os.path.join(PROJ_DIR, 'lists/train_clipfolder.list'),
      read_size=read_size * 1,
      num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
      crop_size=c3d_model.CROP_SIZE,
      shuffle=True
    )
    images_t, labels_t, next_tr, _, _ = read_clip_and_label_v2(
      filename=os.path.join(PROJ_DIR, 'lists/test_clipfolder.list'),
      read_size=read_size * 1,
      num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
      crop_size=c3d_model.CROP_SIZE,
      shuffle=True
    )
  else:
    import collect_im_from_v as cimv
    train_images, train_labels = cimv.collect_train_data(
      c3d_model.NUM_FRAMES_PER_CLIP,c3d_model.INPUT_IMAGE_SIZE,c3d_model.NUM_CLASSES,num_of_clips_pro_class)

    images_t, labels_t = cimv.collect_test_data(
      c3d_model.NUM_FRAMES_PER_CLIP,c3d_model.INPUT_IMAGE_SIZE,c3d_model.NUM_CLASSES,num_of_clips_pro_class=8)
  model = c3d_model.get_model_3l(dropouts,summary=True, backend=backend)
  if 'con' in train_mode:
    model_weight_filename = os.path.join(model_path, model_name)
    if os.path.exists(model_weight_filename):
      model.load_weights(model_weight_filename)
      print 'continued training, following',model_name
    else:
      print 'model not found, start a new training'
  sgd = SGD(lr=learning_rate, momentum=momentum)  #
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  save_params = ModelCheckpoint(filepath=model_path + log_LABEL + '_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val_loss', verbose=2,
                                save_best_only=False, save_weights_only=False, mode='auto')
  model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
            validation_data=(images_t, labels_t), callbacks=[save_params])
  '''
  model.fit(train_images, train_labels, batch_size=batch_size, nb_epoch=epochs, verbose=1, shuffle=True,
                    validation_data=(images_t, masks_t), callbacks=[save_params])
  '''
  model.save_weights('sports1M_weights_custom.h5', overwrite=True)
  json_string = model.to_json()
  with open('sports1M_model_custom_3l_cliplen16.json', 'w') as f:
    f.write(json_string)

def main():
  global train_mode
  train_mode = 'con_train'
  MAX_ITERATION = 10
  learning_rate = 0.00000314#0.000000000015 #0.0000003  # 0.000003
  dropouts = [0.38,0.45,0.5,0.35]
  model_name = 'k_4_0314_05-0.26.hdf5'
  #model_name = 'k_16_0314_03-0.51.hdf5'
  model_path = os.path.join(PROJ_DIR, 'log_models/')
  if not os.path.exists(model_path): 
    os.makedirs(model_path)
  log_LABEL = 'k_4_0314'
  batch_size = 1
  momentum = 0.99
  num_of_clips_pro_class = 180
  train(MAX_ITERATION, learning_rate,dropouts,model_path, model_name,log_LABEL, batch_size, momentum,num_of_clips_pro_class)

if __name__ == '__main__':
  main()
#1,078,294 shallow3,cliplen4
#4,088,854 shallow3,cliplen16
#xxx,xxx,xxx deep3,clip4 1.0228 s
#Train on