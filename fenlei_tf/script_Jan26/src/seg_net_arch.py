from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Reshape, Permute, Activation, \
  Cropping2D
from keras.optimizers import Adam, SGD
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras.utils.visualize_util import plot

smooth = 1.

def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
  return 1-dice_coef(y_true, y_pred)

def testbench_arch(h,w):
  
  inputs = Input((1, h, w)) # 160 x 160
  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(inputs)
  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  pool1 = Dropout(0.15)(pool1)

  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(pool1)
  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  pool2 = Dropout(0.25)(pool2)

  model = Model(input=inputs, output=conv2)
  model.summary()
  #plot(model, "model.png")
  return model

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
  up_1 = merge([UpSampling2D(size=(2, 1))(conv_8), pool5], mode='concat', concat_axis=1)
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

def unet_arch_6c(h, w):
  print("Model of size: %d %d" % (h, w))
  ch = 3
  ordering = 'th' # 'th': (ch, h, w),  'tf': (h, w, ch)
  inputs = Input(shape=(ch, h , w)) # 160 x 160
  #inputs = Input(shape=(h , w,ch)) # 160 x 160

  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(inputs)
  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv1)
  pool1 = Dropout(0.15)(pool1)
  print 'pool1', pool1.get_shape()

  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool1)
  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv2)
  pool2 = Dropout(0.25)(pool2)
  print 'pool2', pool2.get_shape()

  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool2)
  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv3)
  pool3 = Dropout(0.4)(pool3)
  print 'pool3', pool3.get_shape()

  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool3)
  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv4)
  print 'conv4', conv4.get_shape()
  pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv4)
  pool4 = Dropout(0.5)(pool4)
  print 'pool4', pool4.get_shape()

  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool4)
  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv5)
  # pool5 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv5) # 5x5
  # pool5 = Dropout(0.5)(pool5)
  print 'conv5', conv5.get_shape()


  up1 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)
  #print 'up1', up1.get_shape()
  up1 = merge([up1, conv4], mode='concat', concat_axis=1)
  #up1 = merge([(UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)), pool4], mode='concat', concat_axis=1)
  up1 = Dropout(0.4)(up1)
  print 'up1', up1.get_shape()
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up1)
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv8)
  print 'conv8', conv8.get_shape()

  up2 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv8)
  up2 = merge([up2, conv3], mode='concat', concat_axis=1)
  #up2 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
  up2 = Dropout(0.25)(up2)
  print 'up2',up2.get_shape()
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up2)
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv9)
  print 'conv9',conv9.get_shape()  # 7,80,32
  print 'conv2',conv2.get_shape()  # 1,160,16

  up3 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv9)   # 14, 160, 32
  up3 = merge([up3, conv2], mode='concat', concat_axis=1)
  #up3 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
  up3 = Dropout(0.15)(up3)
  print 'up3',up3.get_shape()
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up3)
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv10)
  print 'conv10',conv10.get_shape()

  up4 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv10)
  up4 = merge([up4, conv1], mode='concat', concat_axis=1)
  #up4 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
  up4 = Dropout(0.15)(up4)
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up4)
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv11)

  predictions = Convolution2D(ch, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv11)

  model = Model(input=inputs, output=predictions)
  model.summary()
  #plot(model, "model.png")
  return model


def unet_arch_2c(h, w):
  print("Model of size: %d %d" % (h, w))
  ch = 1 # 1
  inputs = Input(shape=(ch, h , w)) # 160 x 160
  ordering = 'th'  # 'th': (ch, h, w),  'tf': (h, w, ch)

  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(inputs)
  conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv1)
  pool1 = Dropout(0.15)(pool1)
  print 'pool1', pool1.get_shape()

  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool1)
  conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv2)
  pool2 = Dropout(0.25)(pool2)
  print 'pool2', pool2.get_shape()

  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool2)
  conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv3)
  pool3 = Dropout(0.4)(pool3)
  print 'pool3', pool3.get_shape()

  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool3)
  conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv4)
  print 'conv4', conv4.get_shape()
  pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv4)
  pool4 = Dropout(0.5)(pool4)
  print 'pool4', pool4.get_shape()

  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(pool4)
  conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv5)
  # pool5 = MaxPooling2D(pool_size=(2, 2),dim_ordering=ordering)(conv5) # 5x5
  # pool5 = Dropout(0.5)(pool5)
  print 'conv5', conv5.get_shape()


  up1 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)
  #print 'up1', up1.get_shape()
  up1 = merge([up1, conv4], mode='concat', concat_axis=1)
  #up1 = merge([(UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv5)), pool4], mode='concat', concat_axis=1)
  up1 = Dropout(0.4)(up1)
  print 'up1', up1.get_shape()
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up1)
  conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv8)
  print 'conv8', conv8.get_shape()

  up2 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv8)
  up2 = merge([up2, conv3], mode='concat', concat_axis=1)
  #up2 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
  up2 = Dropout(0.25)(up2)
  print 'up2',up2.get_shape()
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up2)
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv9)
  print 'conv9',conv9.get_shape()  # 7,80,32
  print 'conv2',conv2.get_shape()  # 1,160,16

  up3 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv9)   # 14, 160, 32
  up3 = merge([up3, conv2], mode='concat', concat_axis=1)
  #up3 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
  up3 = Dropout(0.15)(up3)
  print 'up3',up3.get_shape()
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up3)
  conv10 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv10)
  print 'conv10',conv10.get_shape()

  up4 = UpSampling2D(size=(2, 2),dim_ordering=ordering)(conv10)
  up4 = merge([up4, conv1], mode='concat', concat_axis=1)
  #up4 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
  up4 = Dropout(0.15)(up4)
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(up4)
  conv11 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='he_normal',dim_ordering=ordering)(conv11)

  predictions = Convolution2D(ch, 1, 1, activation='sigmoid', init='he_normal',dim_ordering=ordering)(conv11)

  model = Model(input=inputs, output=predictions)
  model.summary()
  #plot(model, "model.png")
  return model
