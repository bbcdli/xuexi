#originally by Hamed, 25Apr.2016
#hy: similar to tensor_cnn_video.py

import read_images
import cv2
import numpy as np
import tensorflow as tf
from sklearn import datasets
import math
import imutils

# Train or Evaluation
RETRAIN = False


# Data
#LABELPATH='file_and_label_list'
LABELPATH='FileList.txt'
LABELS = ['Empty/', 'Kiwa/', 'Person/', 'Person+KiWa/', 'Wheelchair/']

# Parameters
learning_rate = 0.003 # TODO 0.05
#learning_rate = 1.5
training_iters = 1500
batch_size = 128
display_step = 1


# Network Parameters
# n_input = 32 * 32  # Cifar data input (img shape: 32*32)
n_input = 24 * 42  # Cifar data input (img shape: 32*32)
n_classes = 5  # Cifar10 total classes (0-9 digits)
dropout = 0.80  # Dropout, probability to keep units
#n_hidden = 200
n_hidden = 45
#n_hidden = 80

# Noise level
noise_level = 0


def confusion_matrix(labels_onehot, scores, normalized=True):
  n_samples, n_class = scores.shape
  conf_matrix = np.zeros((n_class, n_class), dtype=np.float32)

  for i in range(0, n_samples):
    label = np.argmax(labels_onehot[i, :])
    predict = np.argmax(scores[i, :])
    #print label, predict
    #print labels_onehot[i, :], scores[i, :]
    conf_matrix[label, predict] = conf_matrix[label, predict] + 1

  if normalized:
    for i in range(0, n_class):
      conf_matrix[i, :] = conf_matrix[i, :]/np.sum(conf_matrix[i, :])


  return conf_matrix



def dense_to_one_hot(labels_dense, num_classes=n_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  print(labels_one_hot[0])
  return labels_one_hot


# Implementing softmax function on the DL output scores, adopted only for 2 classes
def convert_to_confidence(scores):
  h, w = scores.shape
  output = np.zeros((h,w),dtype=np.float32)
  sum = np.zeros((h, 1), dtype=np.float32)
  for i in range(0, w):
    sum[:,0] = sum[:, 0]+ np.exp(scores[:,i])
  for i in range(0, w):
    output[:,i] = np.exp(scores[:,i])/sum[:, 0]
#    class0=math.exp(scores[0,0])/(math.exp(scores[0,1])+math.exp(scores[0,0]))
#    class1=math.exp(scores[0,1])/(math.exp(scores[0,1])+math.exp(scores[0,0]))
#  output=[class0, class1]
  return output

# Adds noise to gray level images, nomalizes the image again
def add_noise(img,noise_level):
  img=img.astype(np.float32)
  h=img.shape[0]
  w=img.shape[1]
  img_noised=img+np.random.rand(h,w)*noise_level
  img_noised=(img_noised/np.max(img_noised))*255
  #img_noised=img_noised.astype(np.uint8)
  return img_noised


# import data

if RETRAIN:

  digits = datasets.load_digits(n_class=n_classes)

  carimages,cartargets, f = read_images.read_images(LABELPATH)

  print carimages[0]

  #tmp=Image.fromarray(carimages[0],'L')
  #tmp.show()

  #cv.imshow('First Image', carimages[0].astype(np.uint8))
  #cv.waitKey(0)

  carimages=carimages/255-0.5

  print carimages[0]
  print f[0]

  # You havent yet changed the rest, becareful about sizes and etc.

  print('Shape of Original Database',carimages.shape)
  print('Shape of Labels',cartargets.shape)


  digits.images = carimages.reshape((len(carimages), -1))
  digits.images = np.expand_dims(np.array(digits.images), 2).astype(np.float32)
  print digits.images.shape


  digits.target = np.array(cartargets).astype(np.int32)
  digits.target = dense_to_one_hot(digits.target)

  print digits.target


  # Preparing the test image

  test_image=digits.images[7:8]
  test_lables=digits.target[7:8]





# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, 1])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create model
def conv2d(img, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
  return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(_X, _weights, _biases, _dropout):
  # Reshape input picture
  _X = tf.reshape(_X, shape=[-1, 24, 42, 1])  # TODO num channnels change
  # _X = tf.reshape(_X, shape=[-1, 32, 32, 3])  # TODO num channnels change

  # a = np.array(_X[0])
  # print(a.shape)
  # Image._show(Image.fromarray(a, 'RGB'))

  # Convolution Layer
  conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
  # Max Pooling (down-sampling)
  #conv1 = max_pool(conv1, k=1) # TODO return it to K=2
  # Apply Dropout
  conv1 = tf.nn.dropout(conv1, _dropout) # TODO comment it later

  # # Convolution Layer
  conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
  # # Max Pooling (down-sampling)
  conv2 = max_pool(conv2, k=2)
  # # Apply Dropout
  conv2 = tf.nn.dropout(conv2, _dropout) # TODO comment it later!

  # Fully connected layer
  dense1 = tf.reshape(conv2,[-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
  dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))  # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

  # Output, class prediction
  out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
  return out

SEED = 8
# Store layers weight & bias
weights = {
  'wc1': tf.Variable(tf.random_normal([5, 5, 1, 8], stddev=0.1, seed=SEED)),  # 5x5 conv, 1 input, 32 outputs
  'wc2': tf.Variable(tf.random_normal([5, 5, 8, 16], stddev=0.1, seed=SEED)),  # 5x5 conv, 32 inputs, 64 outputs
  'wd1': tf.Variable(tf.random_normal([16 * 24 / 2 * 42 / 2, n_hidden], stddev=0.1, seed=SEED)),  # fully connected, 8*8*64 inputs, 1024 outputs
  # 'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024], stddev=0.1)),  # fully connected, 8*8*64 inputs, 1024 outputs
  'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1, seed=SEED))  # 1024 inputs, 10 outputs (class prediction)
}

biases = {
  'bc1': tf.Variable(tf.random_normal([8])),
  'bc2': tf.Variable(tf.random_normal([16])),
  'bd1': tf.Variable(tf.random_normal([n_hidden])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # TODO change to ADAM

# Evaluate model
# Problem can be here ...
amaxpred = tf.argmax(pred, 1) # Just to check the bug
amaxy = tf.argmax(y, 1) # Just to check for the bug
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Build the summary operation based on the TF collection of Summaries.

# Adding variables to be visualized

summary=tf.scalar_summary('Acc',accuracy)
tf.scalar_summary('Loss',cost)
tf.image_summary('Images',tf.reshape(x, shape=[-1, 24, 42, 1]),max_images=1)


# images after conv1 before max pool
_X = tf.reshape(x, shape=[-1, 20, 20, 1])
conv1 = conv2d(_X, weights['wc1'], biases['bc1'])
tf.image_summary('Conv1 Images', tf.reshape(x, shape=[-1, 24, 42, 1]), max_images=4)

# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)
tf.image_summary('Conv1-Maxpool Images',tf.reshape(x, shape=[-1, 24/2, 42/2, 1]), max_images=4)


tf.image_summary('Images_weights',tf.reshape(weights['wc1'], [-1,5,5,1]),max_images=4)
tf.histogram_summary('Histogram',weights['wc1'])

summary_op = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()

# Creating a saver for the model
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

  if RETRAIN:

    sess.run(init)

    summary_writer = tf.train.SummaryWriter('/tmp/sum7',graph_def=sess.graph_def)

    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
      # Only a part of data base is used for training, the rest is used for validation
      batch_xs, batch_ys = digits.images[0:1000], digits.target[0:1000]
      # Fit training using batch data
      sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
      if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " \
              + "{:.5f}".format(acc)

        batch_xs, batch_ys = digits.images[1001:1398], digits.target[1001:1398]
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        #cpred = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        #predindex = sess.run(amaxpred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        #targetindex = sess.run(amaxy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        print("Validation accuracy:", acc)
        #print("Correct Predictions:", cpred)
        #print("Argmax of Predictions:", predindex)
        #print("Argmax of y:", targetindex)
        output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        #print "Output of NN:", output
        #print "Targets:", batch_ys
        #output = convert_to_confidence(output)
        confMat = confusion_matrix(batch_ys, output, normalized=True)
        print confMat
        #print np.sum(confMat)
        #print output
        #print digits.target[1001:1398]

      summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
      summary_writer.add_summary(summary_str, step)

      # Save the model
      saver.save(sess,save_path='models',global_step=step)

      step += 1
    print "Optimization Finished!"


  else:

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir="")
    print ckpt.model_checkpoint_path
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)



    # Testing

    print ''
    print("Loading an Image and Detection Process")

    # Load the image
    IMAGE_FILE = '/home/hamed/CCTV/WheelChair_Detect/Data/Person+KiWa/2015-12-10_10.49_21.1.cam_81_1_025860.jpg'
    im = cv2.imread(IMAGE_FILE)
    cv2.imshow("Test image", im)

    print("Shape of test image", im.shape)


    im = imutils.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), width=42)

    cv2.imshow("CNN input", im)

    im = np.asarray(im, np.float32)


    # Adding noise to the street image #TODO
  # im=add_noise(im,5)

    # Bluring the image to help detection #TODO
    #im = cv2.GaussianBlur(im,(5,5),0)

    CONF=0.20

    test_image = im
    test_lables = np.zeros((1, n_classes)) # Making a dummy label tp avoid errors

    print im.size

    # Doing something very stupid here, fix it!
    test_image = im.reshape((-1, im.size))
    #print test_image

    test_image = np.expand_dims(np.array(test_image), 2).astype(np.float32)

    # print test_image

    test_image=test_image/255-0.5 # TODO here is tricky,double check wit respect to the formats

    batch_xs, batch_ys = test_image, test_lables
    output = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
    #print("Output for external=",output)
    #print output
    #output = convert_to_confidence(output)
    print output



    cv2.waitKey(0)
    cv2.destroyAllWindows()




# TODO correcting some of the samples, sometimes the window is a bit large
# TODO Consider bigger images for training, details of a car are not clear in small images
# check at first place if you read images correctly, that incorrecr PIL image that appears at the beginning
# check if 0 is nocar or 1 is nocar
# TODO adding noise can help detection, it can also show power of deep learning as compared to other approaches
# TODO adding noise can show power of deep learning as compared to other approaches
# TODO check above again for making sure
# TODO check print of images for /255 and other numerical compatibility
# TODO check making fake bigger images of the database and see if it works
# TODO chek if size of the cars in the street images are appropriate
# TODO try another street image
# TODO adding batch processing ..., researching and reading about batch processing ...
# TODO Histogram normalization or making sure that colors are similar
# TODO change it to corrcet batch mode, but not Tensorflow batch
# TODO add more negative and better negative examples
# TODO make imbalance between negative and positive samples
# TODO consider confidence measure
# TODO blur images!
# TODO blur images!
# TODO Merge rectangle, use aspect ratio to remove false alarms
# TODO use density of detections in order to remove false alarms
# TODO merge rectangles
# TODO use video cues
# TODO Use a few trained network in parallel, they can only be different in terms of initialization, then vote, it significantly reduces false alarms
# Cars are always correctly detectd, but background detection changes ...
# TODO Save models, with a good name

