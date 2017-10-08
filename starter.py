import numpy as np
import tensorflow as tf
from tqdm import tqdm
from convDefs import*

learning_rate = 0.01
training_epochs = 1000
display_step = 1
batch_size = 128

def load_cifar10(path):
    ds = {}
    for i in range(5):
        with open(path+"/data_batch_"+str(i+1), "rb") as db:
             db_set = pickle.load(db)
             ds["data_"+str(i+1)], ds["labels_"+str(i+1)] = reshape_cifar10(db_set[b"data"]), db_set[b"labels"]
    with open(path+"/test_batch", "rb") as tb:
        test_set = pickle.load(tb)
        ds["test_data"], ds["test_labels"] = reshape_cifar10(test_set[b"data"]), test_set[b"labels"]
    # concatenate batches to form training set
    training_data = (ds["data_1"], ds["data_2"], ds["data_3"], ds["data_4"], ds["data_5"])
    training_labels = (ds["labels_1"], ds["labels_2"], ds["labels_3"], ds["labels_4"], ds["labels_5"])
    ds["training_data"] = np.concatenate(training_data, axis=0)
    ds["training_labels"] = np.concatenate(training_labels, axis=0)
    return ds

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
        [input_size, output_size],
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases",
        output_size,
        initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))

def NETWORK(x,input_shape=[32,32,3]):

    with tf.name_scope("NET"):
        with tf.variable_scope("conv1"):
            conv1 = conv_relu(X_pad, kernel_shape=[7, 7, 3, 32], bias_shape=[32])
        with tf.variable_scope("fcl1"):
            output = fcl(tf.reshape(conv1,[-1, 32*32*32]), [32*32*32, 10], [10])
    return tf.nn.softmax(output)


X = tf.placeholder("float", [None] + [32,32,3], name="input")
Y = tf.placeholder("int64", [None], name="labels")
labels = tf.one_hot(Y, 10, axis=-1, name="targets", dtype="int64")

pred = NETWORK(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(50000/batch_size)

    for epoch in range(training_epochs):
        # Loop over all batches
        batch_xs, batch_ys = #Get data
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        if (epoch%display_step==0):
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
            print("Iter " + str(epoch) + ", Minibatch Loss = " + str(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
    print("Optimization Finished!")

    # Calculate accuracy on test data
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: Images, Y: Labels, keep_percent: 1.}))
