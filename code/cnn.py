"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/4232017

Implementation of Convolutional Neural Network
Adapted from:
https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet

TODO: Update CNN class so different network setups can be called via
function. Allows for flexible testing
"""
import os
import time
import random

import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize

from IPython import embed

import preprocessing

max_dims = np.array([0,0,0])
total_pixels = 0
SLICE_COUNT = 20

RESIZE_X = 50  # cols
RESIZE_Y = 50  # rows
RESIZE_Z = 20  # depth

n_classes = 2
batch_size = 10

keep_rate = 0.8

hm_epochs = 10
epoch_size = 1500


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)


def conv3d(x, W, strides=[1, 1, 1, 1, 1]):
    """
    Creates tf node for 3d convolution
    @param x Image to convolve over
    @param W Convolution kernel
    @param strides Convolution strides
    @return tensorflow conv3d node
    """
    return tf.nn.conv3d(x, W, strides=strides, padding='SAME')


def maxpool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1]):
    """
    @param x Data to max pool with
    @param ksize size of window
    @param strides Movement of window during sliding
    @return tensorflow max pool 3d node
    """
    return tf.nn.max_pool3d(x, ksize=ksize, strides=strides, padding='SAME')


def cnn(x):
    """
    Creates tensorflow graph for cnn
    """
    # W_conv1 uses 5x5x5 patches, 1 channel, 32 features
    # W_conv2 uses 5x5x5 patches, 32 channels, 64 features
    # W_fc uses 64 features
    # out combines into 2 output classes (cancer vs. no cancer)
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    # Biases created to match corresponding weight dimensions
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape into [-1, image x, image y, image z, 1])
    # x = tf.reshape(x, shape=[-1, max_dims[0], max_dims[1], max_dims[2], 1])
    x = tf.reshape(x, shape=[-1, RESIZE_X, RESIZE_Y, RESIZE_Z, 1])

    # Create convolutional layers
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    # Fully connected layer with dropout
    # 54080 chosen arbitrarily TODO
    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


if __name__ == "__main__":

    # load data
    DATA_FOLDER = "data\\stage1_preprocessed\\"
    SAMPLE_FOLDER = "data\\sample_preprocessed\\"

    INPUT_FOLDER = DATA_FOLDER
    preprocess_list = os.listdir(INPUT_FOLDER)

    # Online preprocessing steps when data is read:
    #   Normalize data
    #   Zero-center data
    #   Pad 3D images to be equally sized
    # Get max padding bounds before network training
    max_dims = preprocessing.get_max_bounds(INPUT_FOLDER, preprocess_list)
    max_dims[0] = RESIZE_Y
    max_dims[1] = RESIZE_X
    total_pixels = max_dims[0]*max_dims[1]*max_dims[2]

    ######### TRAIN_NEURAL_NETWORK #####
    # TODO put this in function, rest of CNN goes in class
    # to load item, np.load('elem.npy')
    train_data = preprocess_list[:-50]  # All but last 2 elements
    validation_data = preprocess_list[-50:]  # Last 2 elements
    print("Separated training from validation")

    prediction = cnn(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction,
            labels=tf.one_hot(y, n_classes)))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    print("Created graph")

    # setup and initialize session
    config = tf.ConfigProto()
    config.operation_timeout_in_ms = 10000
    sess = tf.InteractiveSession()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)

    successful_runs = 0
    total_runs = 0
    print("Begin training...")
    start_time = time.time()

    for epoch in range(hm_epochs):
        epoch_loss = 0
        epoch_start = time.time()

        # Take a random sample of epoch_size from train_data
        sample_filenames = random.sample(train_data, epoch_size)
        for filename in sample_filenames:
            ############ LOAD AND FORMAT ################
            data = np.load(INPUT_FOLDER + filename)

            # # Temp - resize to RESIZE_X x RESIZE_Y x RESIZE_Z
            data_resized_temp = np.ndarray(
                shape=(RESIZE_Y, RESIZE_X, data[0].shape[2]))
            for slice_num in range(data[0].shape[2]):
                data_resized_temp[..., slice_num] = imresize(
                    data[0][..., slice_num],
                    (RESIZE_Y, RESIZE_X))

            data_resized = np.ndarray(
                shape=(RESIZE_Y, RESIZE_X, RESIZE_Z))
            for i in range(RESIZE_Y):
                data_resized[i, ...] = imresize(
                    data_resized_temp[i, ...],
                    (RESIZE_X, RESIZE_Z))


            data[0] = data_resized
            # print (data[0].shape)
            data[0] = data[0].astype(float)

            # Pad image
            # pad_size = max_dims - data[0].shape
            # data[0] = np.pad(
            #     data[0],
            #     ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])),
            #     'constant')
            # print(data[0].shape)
            # Run normalization, zero-centering

            total_runs += 1

            ############## TRAIN ###############
            # Actual code from tutorial
            try:
                X = data[0]
                Y = data[2]
                _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                epoch_loss += c
                successful_runs += 1
                # print('success', c)
            except Exception as e:
                pass
                # print('failed to read', filename)

        print('Epoch', epoch + 1, 'completed out of',
              hm_epochs, 'loss:', epoch_loss)
        print('Time:', time.time() - epoch_start)

        # ############## VALIDATE #############
        # correct = tf.equal(tf.argmax(prediction, 1), tf.cast(y, tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # for filename in validation_data:

        #     data = np.load(INPUT_FOLDER + filename)

        #     # # Temp - resize to RESIZE_X x RESIZE_Y x RESIZE_Z
        #     data_resized_temp = np.ndarray(
        #         shape=(RESIZE_Y, RESIZE_X, data[0].shape[2]))
        #     for slice_num in range(data[0].shape[2]):
        #         data_resized_temp[..., slice_num] = imresize(
        #             data[0][..., slice_num],
        #             (RESIZE_Y, RESIZE_X))

        #     data_resized = np.ndarray(
        #         shape=(RESIZE_Y, RESIZE_X, RESIZE_Z))
        #     for i in range(RESIZE_Y):
        #         data_resized[i, ...] = imresize(
        #             data_resized_temp[i, ...],
        #             (RESIZE_X, RESIZE_Z))


        #     # print(data[0].shape)
        #     data[0] = data_resized
        #     data[0] = data[0].astype(float)

        #     try:
        #         print('Accuracy', accuracy.eval({
        #             x:data[0],
        #             y:data[2]
        #             }))
        #     except Exception as e:
        #         pass



        # print('Accuracy:',accuracy.eval({
        #     x:[np.load(INPUT_FOLDER + i)[0] for i in validation_data],
        #     y:[np.load(INPUT_FOLDER + i)[2] for i in validation_data]
        #     }))

        # print('Done. Finishing accuracy:')
        # print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:', successful_runs / total_runs)
        print('Total time:', time.time() - start_time)


