import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
# from convDefs import*

learning_rate = 0.01
training_epochs = 10
display_step = 1
batch_size = 128

def reshape_cifar10(imgs_raw):
    imgs_reshaped = np.array(imgs_raw)
    imgs_reshaped = imgs_reshaped.reshape(len(imgs_reshaped), 32, 32, 3)

    return imgs_reshaped

def load_cifar10(path):
    ds = {}
    for i in range(5):
        with open(path+"/data_batch_"+str(i+1), "rb") as db:
             db_set = pickle.load(db)
             ds["data_"+str(i+1)], ds["labels_"+str(i+1)] = reshape_cifar10(db_set[b"data"]), db_set[b"labels"]
    with open(path+"/test_batch", "rb") as tb:
        test_set = pickle.load(tb)
        ds["test_data"], ds["test_labels"] = reshape_cifar10(test_set[b"data"]), test_set[b"labels"]
        ds["data_display_" + str(i + 1)] = np.transpose(np.reshape(ds["data_" + str(i + 1)], (len(ds["data_" + str(i + 1)]), 3, 32, 32)), (0, 2, 3, 1))
        ds["test_data_display"] = np.transpose(np.reshape(ds["test_data"], (len(ds["test_data"]), 3, 32, 32)), (0, 2, 3, 1))

    # concatenate batches to form training set
    training_data = (ds["data_1"], ds["data_2"], ds["data_3"], ds["data_4"], ds["data_5"])
    training_labels = (ds["labels_1"], ds["labels_2"], ds["labels_3"], ds["labels_4"], ds["labels_5"])
    ds["training_data"] = np.concatenate(training_data, axis=0)
    ds["training_labels"] = np.concatenate(training_labels, axis=0)
    return ds

def global_average_pool_6x6(x):
    # average pooling on 6*6 block (full size of the input feature map), for each input (first 1), for each feature map (last 1)
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],
                        strides=[1, 6, 6, 1], padding='SAME')

def conv_relu(x, kernel_shape, bias_shape, stride=1, padding="SAME"):
    # Create variable named "weights".
    weights = tf.get_variable("weights",
        shape=kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases",
        shape=bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, filter=weights,
        strides=[1, stride, stride, 1],
        padding=padding,
        data_format="NHWC")
    return tf.nn.relu(tf.add(conv, biases))

def fcl(x, input_size, output_size, dropout=0.0): #Fully connected layer
    # Create variable named "weights".
    weights = tf.get_variable("weights",
        # [input_size, output_size],
        input_size,
        initializer = tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases",
        output_size,
        initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))

def NETWORK(x, input_shape=[32,32,3]):
    X_pad = None
    with tf.name_scope("NET"):
        with tf.variable_scope("conv1"):
            conv1 = conv_relu(x, kernel_shape=[3, 3, 3, 96], bias_shape=[96], stride=1) # # 3*3 filter, 3 input channel, 96 filters (output channel)
        with tf.variable_scope("conv2"):
            conv2 = conv_relu(conv1, kernel_shape=[3, 3, 96, 96], bias_shape=[96], stride=2) # # 3*3 filter, 96 input channel, 96 filters (output channel)
        with tf.variable_scope("conv3"):
            conv3 = conv_relu(conv2, kernel_shape=[3, 3, 96, 192], bias_shape=[192], stride=2) # # 3*3 filter, 96 input channel, 192 filters (output channel)
        with tf.variable_scope("conv4"):
            conv4 = conv_relu(conv3, kernel_shape=[3, 3, 192, 192], bias_shape=[192], stride=2) # # 3*3 filter, 192 input channel, 192 filters (output channel)

        with tf.variable_scope("conv5"):
            conv5 = conv_relu(conv4, kernel_shape=[3, 3, 192, 192], bias_shape=[192], stride=1) # # 3*3 filter, 192 input channel, 192 filters (output channel)
        with tf.variable_scope("conv6"):
            conv6 = conv_relu(conv5, kernel_shape=[1, 1, 192, 192], bias_shape=[192], stride=1, padding="VALID") # # 1*1 filter, 192 input channel, 192 filters (output channel)
        with tf.variable_scope("conv7"):
            conv7 = conv_relu(conv6, kernel_shape=[1, 1, 192, 10], bias_shape=[10], stride=1, padding="VALID") # # 1*1 filter, 192 input channel, 10 filters (output channel)
        with tf.variable_scope("gap"):
            gap = global_average_pool_6x6(conv7);

        with tf.variable_scope("fcl1"):
            conv4_flat = tf.reshape(conv4, [-1, 32 * 32 * 32])
            output = fcl(conv4_flat, [32*32*32, 10], [10])
            softmax = tf.nn.softmax(output)

        # with tf.variable_scope("softmax"):
        #     softmax = tf.nn.softmax(gap)

    return softmax

def next_batch(imgs, labels, batch, batch_size):
    batch_xs = imgs[batch*batch_size : (batch+1)*batch_size]
    batch_ys = labels[batch * batch_size: (batch + 1) * batch_size]
    return batch_xs, batch_ys


# load data
path = "./cifar-10-batches-py/"
ds = load_cifar10(path)

X = tf.placeholder("float", [None] + [32,32,3], name="input")
Y = tf.placeholder("int64", [None], name="labels")
labels = tf.one_hot(Y, 10, axis=-1, name="targets", dtype="int64")

keep_prob = tf.placeholder("float")

pred = NETWORK(X)
#  tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes,
# and tf.reduce_mean takes the average over these sums.

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

with tf.name_scope('optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost) # train

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(50000/batch_size)

    for epoch in range(training_epochs):
        # Loop over all batches
        print"epoch: ", epoch
        for batch in range(0, total_batch):
            batch_xs, batch_ys = next_batch(ds["training_data"], ds["training_labels"], batch, batch_size) # Get data
            # batch_ys_onehot = tf.one_hot(batch_ys, 10, axis=-1, name="targets", dtype="int64")
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # batch_ys = tf.one_hot(batch_ys, 10, axis=-1, name="targets", dtype="int64") # covert to one-hot
            if (batch % display_step == 0):
                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
                print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss = " + str(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
    print("Optimization Finished!")

    # Calculate accuracy on test data
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: ds["test_data"], Y: ds["test_labels"], keep_prob: 1.}))
