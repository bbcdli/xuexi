from keras.models import Model
from keras.layers import convolutional,Input, merge,Convolution2D, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD
from keras import backend as K
#from keras.utils.visualize_util import plot
#hy12Aug2017 Successfully installed keras-2.0.6 numpy-1.13.1 scipy-0.19.1 theano-0.9.0
#uninstalled after occuring error
#Successfully installed Keras-1.2.2  , (still have numpy-1.13.1 scipy-0.19.1 theano-0.9.0 on)

smooth = 1.
ordering = 'th'
concat_axis = 3

#https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py
def unet_archPaper(h, w):
  print("Model of size: %d %d" % (h, w))

  inputs = Input((1, h , w)) # 160 x 160
  ordering = 'th'  # 'th': (ch, h, w),  'tf': (h, w, ch)

  conv_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(inputs)
  conv_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_1)
  print 'view conv2', conv_2.get_shape()
  pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv_2)
  pool1 = Dropout(0.15)(pool1)
  print 'view pool1', pool1.get_shape()

  conv_3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(pool1)
  conv_4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_3)
  print '\nview conv4', conv_3.get_shape(), '< up-3'
  pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv_4)
  pool2 = Dropout(0.25)(pool2)
  print 'view pool2', pool2.get_shape()

  conv_5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(pool2)
  conv_6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_5)
  print '\nview conv6', conv_5.get_shape(), '< up-2'
  pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv_6)
  pool3 = Dropout(0.4)(pool3)
  print 'view pool3', pool3.get_shape()

  conv_7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(pool3)
  conv_8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_7)
  print '\nview conv8', conv_8.get_shape(), '< up-1'
  pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv_8)
  pool4 = Dropout(0.5)(pool4)
  print 'view pool4', pool4.get_shape()

  conv_9 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(pool4)
  print '\nview conv9', conv_9.get_shape()
  conv_10 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_9)
  print 'view conv10', conv_10.get_shape()
  pool5 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv_10) # 5x5
  pool5 = Dropout(0.5)(pool5)
  print 'view pool5', pool5.get_shape()

  ####################################################################################################
  up_1 = merge([UpSampling2D(size=(2, 2))(conv_8), pool5], mode='concat', concat_axis=1)
  print '\nview up1', up_1.get_shape()
  conv_12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(up_1)
  conv_13 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_12)

  pool6 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv_13)  # 5x5
  pool6 = Dropout(0.5)(pool6)
  print 'view pool6', pool6.get_shape()

  ##################
  up_2 = merge([UpSampling2D(size=(2, 1))(conv_6), pool6], mode='concat', concat_axis=1)
  print '\nview up2', up_2.get_shape()
  conv_15 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(up_2)
  conv_16 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv_15)
  print 'view conv16', conv_16.get_shape()
  pool7 = Dropout(0.15)(conv_16)
  print 'view pool7', pool7.get_shape()

  ##################
  up_3 = merge([UpSampling2D(size=(2, 1))(conv_4), pool7], mode='concat', concat_axis=1)
  print '\nview up3', up_3.get_shape()
  conv_18 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(up_3)
  conv_19 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_18)
  print 'view conv18', conv_18.get_shape()
  pool8 = Dropout(0.4)(conv_19)
  print 'view pool8', pool8.get_shape()

  ##################
  up_4 = merge([UpSampling2D(size=(2, 1))(conv_2), pool8], mode='concat', concat_axis=1)
  print 'view up4', up_4.get_shape()
  conv_21 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(up_4)
  print 'view conv9-1', conv_21.get_shape()
  conv_22 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv_21)
  print 'view conv9', conv_22.get_shape()
  pool9 = Dropout(0.25)(conv_22)
  ##################################################################


  conv_23 = Convolution2D(1, 1, 1, activation='sigmoid', init = 'he_normal')(pool9)
  conv_24 = Convolution2D(1, 1, 1, activation='sigmoid', init = 'he_normal')(conv_23)
  print 'view conv10', conv_24.get_shape()

  model = Model(input=inputs, output=conv_24)
  #model = Model(input=inputs, output=conv12)
  model.summary()
  #plot(model, "model.png")
  return model

#https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py



def segnet_arch_2c(h,w,dropouts):
  
  print("Model of size: %d %d" % (h, w))
  ch = 1
  ordering = 'th' # 'th': (ch, h, w),  'tf': (h, w, ch)
  inputs = Input(shape=(ch, h, w))
  concat_axis = 1
  
  #              0       1      2      3    4     5      6     7      8
  #dropouts =  [0.37,   0.51,  0.34,  0.48,  1,  0.48,  0.28,  0.78,  0.8]
  #dropouts =  [[0.15,0.25,0.4,0.5,1,0.4,0.25,0.15,0.15]]

  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(inputs)
  conv2 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv2)
  pool1 = Dropout(dropouts[0])(pool1)
  print 'pool1', pool1.get_shape()

  conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(pool1)
  conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(conv3)
  pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv4)
  pool2 = Dropout(dropouts[1])(pool2)
  print 'pool2', pool2.get_shape()

  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(pool2)
  conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(conv5)
  pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv6)
  pool3 = Dropout(dropouts[2])(pool3)
  print 'pool3', pool3.get_shape()

  conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(pool3)
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(conv7)
  print 'conv8', conv8.get_shape()
  pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv8)
  pool4 = Dropout(dropouts[3])(pool4)
  print 'pool4', pool4.get_shape()

  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(pool4)
  conv10 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(
    conv9)
  #pool5 = MaxPooling2D(pool_size=(2, 2), dim_ordering=ordering)(conv10)  # 5x5
  #pool5 = Dropout(dropouts[4])(pool5)
  print 'conv10', conv10.get_shape()

  up1 = UpSampling2D(size=(2, 2), dim_ordering=ordering)(conv10)
  print 'up1 upsampling2D:', up1.get_shape()
  up1 = merge([up1, conv8], mode='concat', concat_axis=concat_axis)
  # up1 = merge([(UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)), pool4], mode='concat', concat_axis=1)
  up1 = Dropout(dropouts[5])(up1)
  print 'up1', up1.get_shape()
  conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(up1)
  conv12 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(
    conv11)
  print 'conv12', conv12.get_shape()

  up2 = UpSampling2D(size=(2, 2), dim_ordering=ordering)(conv12)
  print 'up2 upsampling2D:', up2.get_shape()
  up2 = merge([up2, conv6], mode='concat', concat_axis=concat_axis)
  # up2 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
  up2 = Dropout(dropouts[6])(up2)
  print 'up2', up2.get_shape()
  conv13 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(up2)
  conv14 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(
    conv13)
  print 'conv13', conv13.get_shape()  # 7,80,32
  print 'conv2', conv4.get_shape()  # 1,160,16

  up3 = UpSampling2D(size=(2, 2), dim_ordering=ordering)(conv14)  # 14, 160, 32
  print 'up3 upsampling2D:', up3.get_shape()
  up3 = merge([up3, conv4], mode='concat', concat_axis=concat_axis)
  # up3 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
  up3 = Dropout(dropouts[7])(up3)
  print 'up3', up3.get_shape()
  conv15 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(up3)
  conv16 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(
    conv15)
  print 'conv16', conv16.get_shape()

  up4 = UpSampling2D(size=(2, 2), dim_ordering=ordering)(conv16)
  print 'up4 upsampling2D:', up4.get_shape()
  up4 = merge([up4, conv2], mode='concat', concat_axis=concat_axis)
  # up4 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
  up4 = Dropout(dropouts[8])(up4)
  conv17 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(up4)
  conv18 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal', dim_ordering=ordering)(conv17)
  print 'conv18 shape:', conv18.get_shape()
  #predictions = Convolution2D(ch, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv18) #old
  predictions = Convolution2D(ch, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv18) #old

  '''
  dense1 = Flatten()(conv19)
  print 'dense1 shape',dense1.get_shape()
  dense1 = Dropout(1)(dense1)
  
  predictions = Dense(input_dim=ch*1*1,output_dim =h*w,init = 'he_normal',activation = 'softmax')(dense1)
  print 'precision get shape',predictions.get_shape()
  '''
  model = Model(input=inputs, output=predictions)
  model.summary()
  #plot(model, "model.png")
  return model,predictions

def segnet_arch_2c_rgb(h, w):
  print("Model of size: %d %d" % (h, w))
  ch = 3 # 1
  inputs = Input(shape=(ch, h , w))
  ordering = 'th'  # 'th': (ch, h, w),  'tf': (h, w, ch)
  #             0       1      2      3    4     5      6     7      8
  dropouts = [0.37,  0.51,   0.34,  0.48,  1,   0.48, 0.28, 0.78,  0.8]

  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(inputs)
  print 'conv1', conv1.get_shape()
  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv1)
  print 'conv1.', conv1.get_shape()
  pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv1)
  pool1 = Dropout(dropouts[0])(pool1)
  print 'pool1', pool1.get_shape()

  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool1)
  print 'conv2', conv2.get_shape()
  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv2)
  print 'conv2.', conv2.get_shape()
  pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv2)
  pool2 = Dropout(dropouts[1])(pool2)
  print 'pool2', pool2.get_shape()

  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool2)
  print 'conv3', conv3.get_shape()
  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv3)
  print 'conv3.', conv3.get_shape()
  pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv3)
  pool3 = Dropout(dropouts[2])(pool3)  #changed from 0.4 to 0.25
  print 'pool3', pool3.get_shape()

  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool3)
  print 'conv4', conv4.get_shape()
  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv4)
  print 'conv4.', conv4.get_shape()
  pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv4)
  pool4 = Dropout(dropouts[3])(pool4)  #changed from 0.5 to 0.25
  print 'pool4', pool4.get_shape()

  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool4)
  print 'conv5', conv5.get_shape()
  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv5)
  print 'conv5.', conv5.get_shape()


  up1 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)
  print 'up1 upsampling2D:', up1.get_shape()
  up1 = merge([up1, conv4], mode='concat', concat_axis=1)
  #up1 = merge([(UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)), pool4], mode='concat', concat_axis=1)
  up1 = Dropout(dropouts[4])(up1)
  print 'up1 merge conv4', up1.get_shape()
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up1)
  print 'conv8', conv8.get_shape()
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv8)
  print 'conv8.', conv8.get_shape()

  up2 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv8)
  print 'up2 upsampling2D:', up2.get_shape()
  up2 = merge([up2, conv3], mode='concat', concat_axis=1)
  #up2 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
  up2 = Dropout(dropouts[5])(up2)
  print 'up2 merge conv3',up2.get_shape()
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up2)
  print 'conv9',conv9.get_shape()  # 7,80,32
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv9)
  print 'conv9.',conv9.get_shape()  # 7,80,32

  up3 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv9)   # 14, 160, 32
  print 'up3 upsampling2D:', up3.get_shape()
  up3 = merge([up3, conv2], mode='concat', concat_axis=1)
  #up3 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
  up3 = Dropout(dropouts[6])(up3)
  print 'up3 merge conv2',up3.get_shape()
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up3)
  print 'conv10',conv10.get_shape()
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv10)
  print 'conv10.',conv10.get_shape()

  up4 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv10)
  print 'up4 upsampling2D:', up4.get_shape()
  up4 = merge([up4, conv1], mode='concat', concat_axis=1)
  #up4 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
  up4 = Dropout(dropouts[7])(up4)
  print 'up4 merge conv1',up4.get_shape()
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up4)
  print 'conv11',conv11.get_shape()
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv11)
  print 'conv11.',conv11.get_shape()

  #conv12 = Convolution2D(ch, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv11)
  conv12 = Convolution2D(1, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv11)
  print 'out',conv12.get_shape()

  predictions = K.argmax(conv12, axis=1)
  model = Model(input=inputs, output=[conv12])
  
  model.summary()
  #return model
  return model, predictions

import keras.backend as kb
def dice_coef(y_true, y_pred):
  y_true_f = kb.flatten(y_true)
  y_pred_f = kb.flatten(y_pred)
  intersection = kb.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (kb.sum(y_true_f) + kb.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
