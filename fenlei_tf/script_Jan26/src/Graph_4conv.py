import tensorflow as tf
import settings #hy: collection of global variables
settings.set_global()

####################################################### Header begin ################################################
###############################
dropout = [0.3, 0.3, 0.3, 0.5, 0.5] #3,4,5,5
dropout_1s = [1]*len(dropout)
n_hidden = 720  #162*6 # 128
n_classes = len(settings.LABELS)  #hy: adapt to lego composed of 6 classes. Cifar10 total classes (0-9 digits)
n_input = settings.h_resize * settings.w_resize  #hy
#300: horizontal 20%
#360: until 1200 step good, after that test acc remains
#200: start to increase early, 200, but does not increase lot any more
#150, 250, 300, 330, 400: until 70 iter 17%
optimizer_type = 'GD' #'adam' #GD-'gradient.descent',#'ProximalGradientDescent', #'SGD', #'RMSprop'


######################
#GD
learning_rate = 0.05059 #0.03549 #0.04049 #0.03049 #0.015 #0.07297 #0.09568# TODO 0.05  0.005 better, 0.001 good \0.02, 0.13799 to 0.14 good for 6 classes,
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
def define_model():
    # General input for tensorflow
    #hy: Graph input, same placeholders for various architectures
    x = tf.placeholder(tf.float32, [None, n_input, 1], name="x")
    y = tf.placeholder(tf.float32, [None, n_classes], name="y")


    tensor_h = settings.h_resize
    tensor_w = settings.w_resize


    ################################################ Graph 4conv begin

    keep_prob = tf.placeholder(tf.float32, len(dropout), name="keep_prob")
    # tf Graph input
    # hy: define receptive field size
    # optimal setting so far
    filter_size_1 = 11
    filter_size_2 = 5  # 5
    filter_size_3 = 3
    filter_size_4 = 2
    # filter_size_5 = 2

    SEED = 8  # hy: number of filters in conv1  8, 16, 64
    conv2_out = 16
    conv3_out = 32  # hy: 16, 32, 64 outputs  of final conv layer
    conv4_out = 64


    # conv5_out = 128

    # conv_output = 96 #hy: 16, 32, 64 outputs  of final conv layer

    def conv_net(_X, _weights, _biases, _dropout):
        # - INPUT Layer
        # Reshape input picture
        # _X = tf.reshape(_X, shape=[-1, 24, 42, 1])  # TODO num channnels change

        _X = tf.reshape(_X, shape=[-1, settings.h_resize, settings.w_resize, 1])  # hy: use updated proper values for shape
        print 'input tensor', _X.get_shape()
        # _X = tf.reshape(_X, shape=[-1, 32, 32, 3])  # TODO num channnels change

        # a = np.array(_X[0])
        # print(a.shape)
        # Image._show(Image.fromarray(a, 'RGB'))

        ################################
        # - Convolution Layer 1
        k = 4
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'], k)  # 4
        print 'conv1 ( f=', filter_size_1, 'k=', k, ')', conv1.get_shape()

        # Max Pooling (down-sampling)
        k = 2
        conv1 = max_pool(conv1, k)  # TODO return it to K=2
        print 'conv1 max pooling ( k=', k, ')', conv1.get_shape()
        # Apply Dropout
        conv1 = tf.nn.dropout(conv1, _dropout[0])  # TODO comment it later
        print '- dropout ( keep rate', dropout[0], ')', conv1.get_shape()

        ################################
        # - Convolution Layer 2
        k = 1
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'], k)
        print '\nconv2 ( f=', filter_size_2, 'k=', k, ')', conv2.get_shape()
        # # Max Pooling (down-sampling)
        k = 2
        conv2 = max_pool(conv2, k)
        print 'conv2 - max pooling (k=', k, ')', conv2.get_shape()
        # # Apply Dropout
        conv2 = tf.nn.dropout(conv2, _dropout[1])  # TODO comment it later!
        print '- dropout ( keep rate', dropout[1], ')', conv2.get_shape()

        ################################
        k = 1
        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'], k)
        print '\nconv3 ( f=', filter_size_3, 'k=', k, ')', conv3.get_shape()
        k = 2
        conv3 = max_pool(conv3, k)
        print 'conv3 - max pooling ( k=', k, ')', conv3.get_shape()
        conv3 = tf.nn.dropout(conv3, _dropout[2])
        print '- dropout ( keep rate', dropout[2], ')', conv3.get_shape()

        ################################
        k = 1
        conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'], k)
        print '\nconv4 ( f=', filter_size_4, 'k=', k, ')', conv4.get_shape()

        # k = 2
        # conv4 = max_pool(conv4,k)
        # print 'conv4 max pooling ( k=',k,')', conv4.get_shape()

        conv4 = tf.nn.dropout(conv4, _dropout[3])
        print '- dropout ( keep rate', dropout[3], ')', conv4.get_shape()

        '''
        k=1
        conv5 = conv2d(conv4, _weights['wc5'], _biases['bc5'],k)
        print '\nconv5',conv5.get_shape()
        conv5_max = max_pool(conv5,k=2)
        print 'conv5 max p(k=',k,')', conv5.get_shape()
        conv5 = tf.nn.dropout(conv5,_dropout)
        print '- dropout', conv5.get_shape()
        #tf.image_summary('2.Conv', tf.reshape(conv4, shape=[-1, 39, 39, 1]), max_images=2) #hy
        '''
        # Fully connected layer
        dense1 = tf.reshape(conv4,
                            [-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
        print '\ndensel reshape:', dense1.get_shape(), 'hidden_layer', n_hidden
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))  # Relu activation
        print 'densel - relu:', dense1.get_shape()

        dense1 = tf.nn.dropout(dense1, _dropout[4])  # Apply Dropout
        print '- dropout ( keep rate', dropout[4], ')', dense1.get_shape()

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        print 'out:', out.get_shape()
        return out


    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([filter_size_1, filter_size_1, 1, SEED], stddev=0.1, seed=SEED), name="wc1"),
    # 5x5 conv, 1 input, 8 outputs
        'wc2': tf.Variable(tf.random_normal([filter_size_2, filter_size_2, SEED, conv2_out], stddev=0.1, seed=SEED),
                           name="wc2"),  # 5x5 conv, 8 inputs, 16 outputs
        # hy:added new layers
        'wc3': tf.Variable(tf.random_normal([filter_size_3, filter_size_3, conv2_out, conv3_out], stddev=0.1, seed=SEED),
                           name="wc3"),
        'wc4': tf.Variable(tf.random_normal([filter_size_4, filter_size_4, conv3_out, conv4_out], stddev=0.1, seed=SEED),
                           name="wc4"),
        # 'wc5': tf.Variable(tf.random_normal([filter_size_5,filter_size_5,conv4_out,conv5_out],stddev=0.1,seed=SEED),name="wc5"),

        # 'wd1': tf.Variable(tf.random_normal([16 * 24 / 2 * 42 / 2, n_hidden], stddev=0.1, seed=SEED)),  # fully connected, 8*8*64 inputs, 1024 outputs
        # 'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024], stddev=0.1)),  # fully connected, 8*8*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([6 * 6 * conv4_out, n_hidden], stddev=0.1, seed=SEED), name="wd1"),
    # hy: fully connected, 8*8*64 inputs, 1024 outputs
        # 'wd1': tf.Variable(tf.random_normal([conv2_out * tensor_h / 2 * tensor_w / 2, n_hidden], stddev=0.1, seed=SEED),name="wd1"), #hy: fully connected, 8*8*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1, seed=SEED), name="w_out")
    # 1024 inputs, 10 outputs (class prediction)
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([SEED]), name="bc1"),
        'bc2': tf.Variable(tf.random_normal([conv2_out]), name="bc2"),  # hy: use variable, instead fixed number
        'bc3': tf.Variable(tf.random_normal([conv3_out]), name="bc3"),
        'bc4': tf.Variable(tf.random_normal([conv4_out]), name="bc4"),
        'bd1': tf.Variable(tf.random_normal([n_hidden]), name="bd1"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="b_out")  # hy: predict probability of each class at output
    }

    # hy: try with zero mean
    # tf.image.per_image_whitening(x)
    # this operation computes (x-mean)/adjusted_stddev

    pred = conv_net(x, weights, biases, keep_prob)

    pred = tf.add(pred, 0, name="pred")
    ############################################################
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y), name="cost")

    # learning_rate = tf.train.exponential_decay(
    #  0.01,                # Base learning rate.
    #  batch * BATCH_SIZE,  # Current index into the dataset.
    #  train_size,          # Decay step.
    #  0.95,                # Decay rate.
    #  staircase=True)

    if optimizer_type == 'adam':
        # hy: Adam with these parameters beta1=0.9,beta2=0.999, epsilon=1e-08 etc the training accuracy is not stable, epsilon = 0.01 better for these data
        print '\noptimizer:', optimizer_type, 'learning_rate:', learning_rate, '\nbeta11:', beta1, '\tbeta2:', beta2, '\tepsilon:', epsilon
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
                                           use_locking=False, name='Adam').minimize(cost)

    # hy: Adam with only learning rate as parameter can also be used to continue a training that was done previously with beta,epsilon setup
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # TODO change to ADAM

    if optimizer_type == 'GD':
        # hy: GradientDescentOptimizer
        print '\noptimizer:', optimizer_type, '\tlearning_rate:', learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="GD").minimize(cost)
        # for learning_rate_i in xrange(5):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_i*(1+learning_rate_i*0.000001),name="GD").minimize(cost)

    # hy: Adam with only learning rate as parameter can also be used to continue a training that was done previously with beta,epsilon setup
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # TODO change to ADAM

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Use simple momentum for the optimization.hy: it is an optimizer subclass, and is used after Gradients are processed
    # optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=batch_size)

    # Evaluate model
    # Problem can be here ...
    amaxpred = tf.argmax(pred, 1)  # Just to check the bug
    amaxy = tf.argmax(y, 1)  # Just to check for the debug
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Build the summary operation based on the TF collection of Summaries.

    # Adding variables to be visualized
    # hy:add diagrams
    summary = tf.scalar_summary('Accuracy', accuracy)
    tf.scalar_summary('Loss', cost)

    # tf.image_summary('Images Original',tf.reshape(x, shape=[-1, 24, 42, 1]),max_images=4)
    tf.image_summary('Original', tf.reshape(x, shape=[-1, tensor_h, tensor_w, 1]), max_images=1)  # hy:images_view

    # images after conv1 before max pool
    # _X = tf.reshape(x, shape=[-1, 24, 42, 1])

    # hy:here devided by number of training size
    _X = tf.reshape(x, shape=[-1, tensor_h, tensor_w, 1])  # hy

    conv_view_size = 46
    ####################################### conv layer image
    # conv1 = tf.placeholder(tf.float32,name="conv1")
    conv1 = conv2d(_X, weights['wc1'], biases['bc1'], 4)  # 4
    conv1 = tf.add(conv1, 0, name="conv1")  # hy
    conv1_size = conv1.get_shape()
    print 'for conv1 view', conv1_size  # hy: define shape can only use constant, not value of get_shape
    tf.image_summary('1.Conv', tf.reshape(conv1, shape=[-1, conv_view_size, conv_view_size, 1]), max_images=4,
                     name="conv1-valid")  # hy
    # tf.image_summary('1.Conv', tf.reshape(conv1, shape=[-1, tensor_h, tensor_w, 1]), max_images=4,name="conv1-same") #hy

    ####################################### conv layer image
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 1)
    conv2 = tf.add(conv2, 0, name="conv2")  # hy conv2
    conv2_size = conv2.get_shape()
    print 'for conv2 view', conv2_size
    # tf.image_summary('Output of Second Convolution',tf.reshape(conv2, shape=[-1, 24, 42, 1]), max_images=16)

    tf.image_summary('2.Conv', tf.reshape(conv2, shape=[-1, conv_view_size, conv_view_size, 1]), max_images=4,
                     name="conv2-valid")  # hy
    # tf.image_summary('2.Conv', tf.reshape(conv2, shape=[-1, tensor_h, tensor_w, 1]), max_images=conv2_out,name="conv2-same") #hy


    ####################################### conv layer image
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 1)
    conv3 = tf.add(conv3, 0, name="conv3")  # hy
    conv3_size = conv3.get_shape()
    print 'for conv3 view', conv3_size
    tf.image_summary('3.Conv', tf.reshape(conv3, shape=[-1, conv_view_size, conv_view_size, 1]), max_images=4,
                     name="conv3-valid")  # hy
    # tf.image_summary('3.Conv', tf.reshape(conv3, shape=[-1, tensor_h, tensor_w, 1]), max_images=4,name="conv3-same") #hy


    ####################################### conv layer image
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 1)
    conv4 = tf.add(conv4, 0, name="conv4")  # hy
    print 'for conv4 view', conv4.get_shape()
    tf.image_summary('4.Conv', tf.reshape(conv4, shape=[-1, conv_view_size, conv_view_size, 1]), max_images=4,
                     name="conv4-valid")  # hy
    # tf.image_summary('4.Conv', tf.reshape(conv4, shape=[-1, 31, 31, 1]), max_images=4,name="conv4-valid") #hy


    ##############
    tf.histogram_summary('Histogram weight 1.Conv', weights['wc1'], name="histogram_conv1")
    tf.histogram_summary('Histogram pred', pred, name="histogram_pred")  # hy: added

    summary_op = tf.merge_all_summaries()


    return (n_hidden,learning_rate,dropout,dropout_1s,optimizer_type,pred,x,y, keep_prob,optimizer,accuracy,cost,summary_op)