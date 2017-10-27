from __future__ import print_function
from __future__ import division
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import *
from LSUV import *
import argparse
import pandas
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import pickle
import os
import math
import uuid

# import cv2
import numpy as np

class AllCNN(Sequential):
    """AllCNN encapsulates the All-CNN.
    """

    def __init__(self, is_dropout = True, is_bn = False, seed = 24, initializer = "glorot_uniform", is_init_fixed = True):
        Sequential.__init__(self)
        self.seed = seed

        # build the network architecture
        if initializer != "LSUV":
            self.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(32, 32, 3), kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(96, 3, 3, border_mode='same', kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), kernel_initializer=initializer))
            if is_dropout:
                self.add(Dropout(0.5))

            self.add(Convolution2D(192, 3, 3, border_mode='same', kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 3, 3, border_mode='same', kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2, 2), kernel_initializer=initializer))
            if is_dropout:
                self.add(Dropout(0.5))

            self.add(Convolution2D(192, 3, 3, border_mode='same', kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 1, 1, border_mode='valid', kernel_initializer=initializer))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(10, 1, 1, border_mode='valid', kernel_initializer=initializer))

            self.add(GlobalAveragePooling2D())
            self.add(Activation('softmax'))
        else:
            self.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(96, 3, 3, border_mode='same'))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2)))
            if is_dropout:
                self.add(Dropout(0.5))

            self.add(Convolution2D(192, 3, 3, border_mode='same'))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 3, 3, border_mode='same'))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2, 2)))
            if is_dropout:
                self.add(Dropout(0.5))

            self.add(Convolution2D(192, 3, 3, border_mode='same'))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(192, 1, 1, border_mode='valid'))
            if is_bn:
                self.add(BatchNormalization())
            self.add(Activation('relu'))
            self.add(Convolution2D(10, 1, 1, border_mode='valid'))

            self.add(GlobalAveragePooling2D())
            self.add(Activation('softmax'))

class LossAccEveryBatch(Callback):
    def on_train_begin(self, logs={}):
        self.losses_batch = []
        self.accs_batch = []

    def on_batch_end(self, batch, logs={}):
        self.losses_batch.append(logs.get('loss'))
        self.accs_batch.append(logs.get('acc'))

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", dest="id", default=str(uuid.uuid4()), type=str,
                        help='the running instance id')

    parser.add_argument("-batchsize", dest="batch_size", default=32, type=int,
                        help='batch size')

    parser.add_argument("-epochs", dest="epochs", default=350, type=int,
                        help='the numer of epochs')

    parser.add_argument("-init", dest="initializer", default="LSUV", type=str,
                        help='the weight initializer')

    parser.add_argument("-retrain", dest="retrain", default=False, type=bool,
                        help='whether to train from the beginning or read weights from the pretrained model')

    parser.add_argument("-weightspath", dest="weights_path", default="all_cnn_weights_0.9088_0.4994.hdf5", type=str,
                        help='the path of the pretrained model/weights')

    parser.add_argument("-train", dest="is_training", action='store_true', default=False,
                        help="whether to train or test")

    parser.add_argument("-bn", dest="is_bn", action='store_true', default=False,
                        help="whether to perform batch normalization")

    parser.add_argument("-dropout", dest="is_dropout", action='store_true', default=False,
                        help="whether to perform dropout with 0.5")

    parser.add_argument("-f", dest="is_plot", action='store_true', default=False,
                        help="whether to plot the figure")


    args = parser.parse_args()

    # encironment configuration
    K.set_image_dim_ordering('tf')
    # matplotlib.use('Agg') # for server using plt

    classes = 10

    # parameters
    id = args.id
    initializer = args.initializer
    batch_size = args.batch_size
    epochs = args.epochs
    retrain = args.retrain
    is_training = args.is_training
    is_bn = args.is_bn
    is_dropout = args.is_dropout

    old_weights_path = args.weights_path
    new_best_weights_path = id + "/" + "all_cnn_best_weights_" + id + ".hdf5"
    whole_model_path = id + "/" + "all_cnn_whole_model_" + id + ".h5"
    history_path = id + "/" + "all_cnn_history_" + id + ".csv"

    accs_epoch_path = id + "/" + "all_cnn_accs_epoch_" + id + ".acc"
    losses_epoch_path = id + "/" + "all_cnn_losses_epoch_" + id + ".loss"
    val_accs_epoch_path = id + "/" + "all_cnn_val_accs_epoch_" + id + ".acc"
    val_losses_epoch_path = id + "/" + "all_cnn_val_losses_epoch_" + id + ".acc"

    accs_batch_path = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
    losses_batch_path = id + "/" + "all_cnn_losses_batch_" + id + ".loss"

    size = 50000
    acc_figure_path = "acc_" + id + ".png"
    loss_figure_path = "loss_" + id + ".png"

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
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
    )

    datagen_test = ImageDataGenerator(
        featurewise_center=False,
        # set input mean to 0 over the dataset (featurewise subtract the mean image from every image in the dataset)
        samplewise_center=False,  # set each sample mean to 0 (for each image each channel)
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False # apply ZCA whitening)
    )

    # initialize the model
    model = AllCNN(is_dropout, is_bn)

    # set training mode
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsp = RMSprop(lr=0.001, rho=0.0, epsilon=1e-08, decay=0.001)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    if is_training:
        # create the directory for the experiment
        if not os.path.exists(id):
            print(id + " directory is created")
            os.makedirs(id)

        datagen_train.fit(X_train) # compute the internal data stats
        datagen_test.fit(X_test)
        if not retrain:
            # load pretrainied model
            print("read weights from the pretrained")
            model.load_weights(old_weights_path)
        else:
            if initializer == "LSUV":
                # initialize the model using LSUV
                print("retrain the model")
                # training_data_shuffled, training_labels_oh_shuffled = shuffle(X_train, Y_train)
                # batch_xs_init = training_data_shuffled[0:batch_size]

                for x_batch, y_batch in datagen_train.flow(X_train, Y_train, batch_size=batch_size): # make use of image processing utility provided by ImageDataGenerator
                    LSUV_init(model, x_batch)
                    break

        print("start training")

        # initialize the callbacks

        # save the best model after every epoch
        checkpoint = ModelCheckpoint(new_best_weights_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                     mode='max')

        # # print the btach number every batch
        # batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))

        # learning schedule callback
        lrate = LearningRateScheduler(step_decay)


        lossAcc = LossAccEveryBatch()
        callbacks_list = [checkpoint, lossAcc]

        # fit the model on the batches generated by datagen.flow()
        # it is real-time data augmentation
        history_callback = model.fit_generator(datagen_train.flow(X_train, Y_train,batch_size=batch_size),
                                               steps_per_epoch=X_train.shape[0]/batch_size,
                                               epochs=epochs, validation_data=datagen_test.flow(X_test, Y_test, batch_size=batch_size), callbacks=callbacks_list,
                                               verbose=1, validation_steps=X_train.shape[0]/batch_size)

        pandas.DataFrame(history_callback.history).to_csv(history_path)
        model.save(whole_model_path)

        # get the stats and dump them for each epoch
        accs_epoch = history_callback.history['acc']
        with open(accs_epoch_path, "w") as fp:  # pickling
            pickle.dump(accs_epoch, fp)

        val_accs_epoch = history_callback.history['val_acc']
        with open(val_accs_epoch_path, "w") as fp:  # pickling
            pickle.dump(val_accs_epoch, fp)

        losses_epoch = history_callback.history['loss']
        with open(losses_epoch_path, "w") as fp:  # pickling
            pickle.dump(losses_epoch, fp)

        val_losses_epoch = history_callback.history['val_loss']
        with open(val_losses_epoch_path, "w") as fp:  # pickling
            pickle.dump(val_losses_epoch, fp)

        # get the stats and dump them for each match
        accs_batch = lossAcc.accs_batch
        with open(accs_batch_path, "w") as fp:  # pickling
            pickle.dump(accs_batch, fp)

        losses_batch = lossAcc.losses_batch
        with open(losses_batch_path, "w") as fp:  # pickling
            pickle.dump(losses_batch, fp)

        # # summarize history for accuracy
        # plt.plot(history_callback.history['acc'])
        # plt.plot(history_callback.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # # plt.show()
        # # plt.savefig(acc_figure_path)
        #
        # # summarize history for loss
        # plt.plot(history_callback.history['loss'])
        # plt.plot(history_callback.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # # plt.show()
        # # plt.savefig(loss_figure_path)

    else:
        print("read weights from the pretrained: ", old_weights_path)
        model.load_weights(old_weights_path)

        datagen_test.fit(X_test)  # compute the internal data stats
        # loss, acc = model.evaluate_generator(datagen_test.flow(X_test, Y_test,batch_size=batch_size),
        #                                        steps_per_epoch=X_test.shape[0]/batch_size,
        #                                        epochs=epochs, verbose=1)

        loss, acc = model.evaluate_generator(datagen_test.flow(X_test, Y_test,batch_size=batch_size),
                                               steps = X_test.shape[0]/batch_size)
        print("loss: ", loss)
        print("acc: ", acc)

if __name__ == "__main__":
    main()