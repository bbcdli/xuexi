#hy: this file contains graph for 3conv with one size dropout rate, can used for test or further training an old model

import tensorflow as tf
import settings #hy: collection of global variables
settings.set_global()

####################################################### Header begin ################################################
dropout = [0.25] #3,4,5,5
dropout_1s = [1]*len(dropout)
n_hidden = 720  #162*6 # 128
n_classes = len(settings.LABELS)  #hy: adapt to lego composed of 6 classes. Cifar10 total classes (0-9 digits)
n_input = settings.h_resize * settings.w_resize  #hy
#300: horizontal 20%
#360: until 1200 step good, after that test acc remains
#200: start to increase early, 200, but does not increase lot any more
#150, 250, 300, 330, 400: until 70 iter 17%
optimizer_type = 'GD' #'adam' #GD-'gradient.descent',#'ProximalGradientDescent', #'SGD', #'RMSprop'
trained_model = "./testbench/" + "model_GD720_h184_w184_c6_3conv_L0.75_R0.65_V1.0_8_0.81-6361.meta"


######################
#GD
learning_rate = 0.04149 #0.03549 #0.04049 #0.03049 #0.015 #0.07297 #0.09568# TODO 0.05  0.005 better, 0.001 good \0.02, 0.13799 to 0.14 good for 6 classes,
######################
#adam
beta1 = 0.9
beta2 = 0.999
epsilon = 0.009
######################
#RMSprop
decay=0.00009
momentum=0
epsilon_R=0.009
######################
#SGD
lr_decay = 0.01
decay_step = 100
######################
####################################################### Header End#### ################################################


# Create model
def conv2d(img, w, b, k):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'), b))

def max_pool(img, k):
  return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


###################################################################
# General input for tensorflow
#hy: Graph input, same placeholders for various architectures
x = tf.placeholder(tf.float32, [None, n_input, 1], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")

keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout (keep probability)

tensor_h = settings.h_resize
tensor_w = settings.w_resize

################################################ Graph_2conv begin
# tf Graph input
# hy: define patch size
filter_size_1 = 5
filter_size_2 = 3
SEED = 8  # hy: 8, 16, 64
conv_output = 16  # hy: 32, 64 outputs  of final conv layer

def conv_net(_X, _weights, _biases, _dropout):

    # - INPUT Layer
    # Reshape input picture
    #_X = tf.reshape(_X, shape=[-1, 24, 42, 1])  # TODO num channnels change
    _X = tf.reshape(_X, shape=[-1, settings.h_resize, settings.w_resize, 1])  #hy: use updated proper values for shape
    # _X = tf.reshape(_X, shape=[-1, 32, 32, 3])  # TODO num channnels change

    # a = np.array(_X[0])
    # print(a.shape)
    # Image._show(Image.fromarray(a, 'RGB'))


    # - Convolution Layer 1
    k = 1
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'], k)
    # Max Pooling (down-sampling)
    k = 2
    conv1 = max_pool(conv1, k) # TODO return it to K=2
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout) # TODO comment it later


    # - Convolution Layer 2
    k = 1
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'], k)
    #k = 2
    # # Max Pooling (down-sampling)
    #conv2 = max_pool(conv2, k=2)
    # # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout) # TODO comment it later!



    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out



# Store layers weight & bias
weights = {
'wc1': tf.Variable(tf.random_normal([filter_size_1, filter_size_1, 1, SEED], stddev=0.1, seed=SEED)),  # 5x5 conv, 1 input, 8 outputs
'wc2': tf.Variable(tf.random_normal([filter_size_2, filter_size_2, SEED, conv_output], stddev=0.1, seed=SEED)),  # 5x5 conv, 8 inputs, 16 outputs

#'wd1': tf.Variable(tf.random_normal([16 * 24 / 2 * 42 / 2, n_hidden], stddev=0.1, seed=SEED)),  # fully connected, 8*8*64 inputs, 1024 outputs
'wd1': tf.Variable(tf.random_normal([conv_output * settings.h_resize / 2 * settings.w_resize / 2, n_hidden], stddev=0.1, seed=SEED)), #hy: fully connected, 8*8*64 inputs, 1024 outputs

# 'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024], stddev=0.1)),  # fully connected, 8*8*64 inputs, 1024 outputs
'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1, seed=SEED))  # 1024 inputs, 10 outputs (class prediction)
}

biases = {
'bc1': tf.Variable(tf.random_normal([SEED])),
'bc2': tf.Variable(tf.random_normal([conv_output])), #hy: use variable, instead fixed number
'bd1': tf.Variable(tf.random_normal([n_hidden])),
'out': tf.Variable(tf.random_normal([n_classes]))  #hy: predict probability of each class at output
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred = tf.add(pred,0,name="pred")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))


'''
learning_rate = tf.train.exponential_decay(
0.01,                # Base learning rate.
batch * BATCH_SIZE,  # Current index into the dataset.
train_size,          # Decay step.
0.95,                # Decay rate.
staircase=True)
'''

# Use simple momentum for the optimization.
#optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                     0.9).minimize(cost,
#                                                   global_step=batch_size)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # TODO change to ADAM

#hy: GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# Problem can be here ...
amaxpred = tf.argmax(pred, 1) # Just to check the bug
amaxy = tf.argmax(y, 1) # Just to check for the bug
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Build the summary operation based on the TF collection of Summaries.

# Adding variables to be visualized

summary = tf.scalar_summary('Acc_Validation', accuracy)
tf.scalar_summary('Loss_Validation', cost)

images_view = 6 #hy number of images to view in tensorflow
#tf.image_summary('Images Original',tf.reshape(x, shape=[-1, 24, 42, 1]),max_images=4)
tf.image_summary('Original', tf.reshape(x, shape=[-1, settings.h_resize, settings.w_resize, 1]), max_images=images_view)#hy

# images after conv1 before max pool
#_X = tf.reshape(x, shape=[-1, 24, 42, 1])
_X = tf.reshape(x, shape=[-1, settings.h_resize, settings.w_resize, 1]) #hy

k = 1
conv1 = conv2d(_X, weights['wc1'], biases['bc1'], k)
#tf.image_summary('Output of First Convolution', tf.reshape(x, shape=[-1, 24, 42, 1]), max_images=4)
#tf.image_summary('Output of First Convolution', tf.reshape(conv1, shape=[-1, 24, 42, 1]), max_images=8)
tf.image_summary('1.Conv', tf.reshape(conv1, shape=[-1, settings.h_resize, settings.w_resize, 1]), max_images=images_view) #hy

print 'tensor matrix shape - conv1'
print conv1

# Max Pooling (down-sampling)
k = 1
conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], k)
#tf.image_summary('Output of Second Convolution',tf.reshape(conv2, shape=[-1, 24, 42, 1]), max_images=16)
tf.image_summary('2.Conv', tf.reshape(conv2, shape=[-1, settings.h_resize, settings.w_resize, 1]), max_images=images_view) #hy

tf.image_summary('Weights 1.Conv', tf.reshape(weights['wc1'], [-1, filter_size_1, filter_size_1, 1]), max_images=images_view)  #hy: use defined var patch
#tf.image_summary('Weights Second Conv', tf.reshape(weights['wc2'], [-1, filter_size_1, filter_size_1, 1]), max_images=8)  #hy: use defined var patch

tf.histogram_summary('Histogram 1.Conv', weights['wc1'])
tf.histogram_summary('Histogram pred', pred, name="histogram_pred")  # hy: added

summary_op = tf.merge_all_summaries()

################################################ Graph_2conv end

