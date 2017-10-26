from __future__ import print_function
from __future__ import division
import tensorflow as tf
import h5py
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from shuffle import *
from LSUV import *
import argparse
import pandas
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl

# import cv2
import numpy as np

class AllCNN(Sequential):
    """AllCNN encapsulates the All-CNN.
    """

    def __init__(self, seed = 24, is_init_fixed = True):
        Sequential.__init__(self)
        self.seed = seed

        # build the network architecture
        self.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
        self.add(Activation('relu'))
        self.add(Convolution2D(96, 3, 3, border_mode='same'))
        self.add(Activation('relu'))
        self.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2)))
        self.add(Dropout(0.5))

        self.add(Convolution2D(192, 3, 3, border_mode='same'))
        self.add(Activation('relu'))
        self.add(Convolution2D(192, 3, 3, border_mode='same'))
        self.add(Activation('relu'))
        self.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2, 2)))
        self.add(Dropout(0.5))

        self.add(Convolution2D(192, 3, 3, border_mode='same'))
        self.add(Activation('relu'))
        self.add(Convolution2D(192, 1, 1, border_mode='valid'))
        self.add(Activation('relu'))
        self.add(Convolution2D(10, 1, 1, border_mode='valid'))

        self.add(GlobalAveragePooling2D())
        self.add(Activation('softmax'))

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

# def main():
parser = argparse.ArgumentParser()
parser.add_argument("-batchsize", dest="batch_size", default=32, type=int,
                    help='batch size')

parser.add_argument("-epoches", dest="epoches", default=350, type=int,
                    help='the numer of epoches')

parser.add_argument("-retrain", dest="retrain", default=False, type=bool,
                    help='whether to train from the benginning or read weights from the pretrained model')

parser.add_argument("-weightspath", dest="weights_path", default="keras_allconv_LSUV.hdf5", type=str,
                    help='weights path')

parser.add_argument("-train", dest="is_training", action='store_true', default=False,
                    help="whether to train or test")
args = parser.parse_args()

# encironment configuration
K.set_image_dim_ordering('tf')
# matplotlib.use('Agg') # for server using plt

classes = 10

# parameters
# batch_size = args.batch_size
# epoches = args.epoches
# retrain = args.retrain
# is_training = args.is_training
# weights_path = args.weights_path

batch_size = 32
epoches = 20
retrain = True
is_training = True
id = "2"
old_weights_path = "keras_allconv_LSUV.hdf5"
new_best_weights_path = "all_cnn_best_weights_" + id + ".hdf5"
new_final_weights_path = "all_cnn_final_weights_" + id + ".h5"
history_path = "all_cnn_history" + id + ".csv"
size = 50000
acc_path = "acc_" + id + ".png"
loss_path = "loss_" + id + ".png"


# load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train[0:size]
Y_train = Y_train[0:size]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train.shape[1:])

Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)

# normalize the images
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# image preprocessing
datagen_train = ImageDataGenerator(
    featurewise_center=False,
    # set input mean to 0 over the dataset (featurewise subtract the mean image from every image in the dataset)
    samplewise_center=False,  # set each sample mean to 0 (for each image each channel)
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)

datagen_test = ImageDataGenerator(
    featurewise_center=False,
    # set input mean to 0 over the dataset (featurewise subtract the mean image from every image in the dataset)
    samplewise_center=False,  # set each sample mean to 0 (for each image each channel)
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True # apply ZCA whitening)
)

# initialize the model
model = AllCNN()

# set training mode
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

if is_training:
    datagen_train.fit(X_train) # compute the internal data stats
    datagen_test.fit(X_test)
    if not retrain:
        # load pretrainied model
        print("read weights from the pretrained")
        model.load_weights(old_weights_path)
    else:
        # initialize the model using LSUV
        print("retrain the model")
        # training_data_shuffled, training_labels_oh_shuffled = shuffle(X_train, Y_train)
        # batch_xs_init = training_data_shuffled[0:batch_size]

        for x_batch, y_batch in datagen_train.flow(X_train, Y_train, batch_size=batch_size): # make use of image processing utility provided by ImageDataGenerator
            LSUV_init(model, x_batch)
            break

    print("start training")

    # save the best model after every epoch
    checkpoint = ModelCheckpoint(new_best_weights_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='max')

    callbacks_list = [checkpoint]

    # fit the model on the batches generated by datagen.flow()
    # it is real-time data augmentation
    history_callback = model.fit_generator(datagen_train.flow(X_train, Y_train,batch_size=batch_size),
                                           steps_per_epoch=X_train.shape[0]/batch_size,
                                           epochs=epoches, validation_data=datagen_test.flow(X_test, Y_test, batch_size=batch_size), callbacks=callbacks_list,
                                           verbose=1, validation_steps=X_train.shape[0]/batch_size)

    # im = cv2.resize(cv2.imread('image.jpg'), (224, 224)).astype(np.float32)
    # out = model.predict(im)
    # print
    # np.argmax(out)
    #
    pandas.DataFrame(history_callback.history).to_csv(history_path)
    model.save(new_final_weights_path)

    # summarize history for accuracy
    plt.plot(history_callback.history['acc'])
    plt.plot(history_callback.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plt.savefig(acc_path)

    # summarize history for loss
    plt.plot(history_callback.history['loss'])
    plt.plot(history_callback.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plt.savefig(loss_path)

else:
    print("read weights from the pretrained")
    model.load_weights(old_weights_path)

    datagen_test.fit(X_test)  # compute the internal data stats
    loss, acc = model.evaluate_generator(datagen_test.flow(X_test, Y_test,batch_size=batch_size),
                                           steps_per_epoch=X_train.shape[0]/batch_size,
                                           epochs=epoches, verbose=1)
    print("loss: ", loss)
    print("acc: ", acc)

# if __name__ == "__main__":
#     main()