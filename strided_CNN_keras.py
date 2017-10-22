from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
# import pandas
# import cv2
import numpy as np

import numpy as np
from read_cifar10 import *
from shuffle import *
from next_batch import *
from LSUV import *

batch_size = 128
num_classes = 10
learning_rate = 0.01
training_epochs = 10
display_step = 1
total_batch = int(50000/batch_size)
seed = 1
np.random.seed(seed)


# load data
path = "./cifar-10-batches-py/"
ds = load_cifar10(path)
# data1 = ds["data_1"]
# labels1 = ds["labels_1"]
# data1_shuffled, labels1_shuffled = shuffle(data1, labels1)

# convert to onehot
ds["training_labels_oh"] = np_utils.to_categorical(ds["training_labels"], num_classes)
ds["test_labels_oh"] = np_utils.to_categorical(ds["test_labels"], num_classes)

# initialize the model
model = Sequential()

model.add(Convolution2D(96, 3, 3, border_mode = 'same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))

model.add(Convolution2D(96, 3, 3,border_mode='same', subsample = (2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(192, 3, 3,border_mode='same', subsample = (2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(192, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))

model.add(Convolution2D(192, 1, 1,border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(10, 1, 1, border_mode='valid'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=learning_rate, rho=0.0, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

# # initialize the model using LSUV
# training_data_shuffled, training_labels_oh_shuffled = shuffle(ds["training_data"], ds["training_labels_oh"])
# training_data_shuffled_normalized = training_data_shuffled.astype('float32')
# training_data_shuffled_normalized = training_data_shuffled_normalized / 255.0  # normalized
# batch_xs_init = training_data_shuffled_normalized[0:batch_size]
# LSUV_init(model, batch_xs_init)

# print summary
print (model.summary())

for epoch in range(training_epochs):
    print ("epoch ", epoch)
    training_data_shuffled, training_labels_oh_shuffled = shuffle(ds["training_data"], ds["training_labels_oh"])
    training_data_shuffled_normalized= training_data_shuffled.astype('float32')
    training_data_shuffled_normalized = training_data_shuffled_normalized / 255.0 # normalized
    # for batch in range(0, total_batch):
    #     batch_xs, batch_ys_oh = next_batch(training_data_shuffled_normalized, training_labels_oh_shuffled, batch, batch_size)  # Get data
    #     [loss, acc] = model.train_on_batch(batch_xs, batch_ys_oh)
    #
    #     print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss = " + str(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
    #
    #     # if (batch % display_step == 0):
    #     #     print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss = " + str(
    #     #         loss) + ", Training Accuracy = " + "{:.5f}".format(acc))

    model.fit(training_data_shuffled, training_labels_oh_shuffled, epochs=1, batch_size=batch_size)