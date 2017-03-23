"""
Author: Neil Jassal
Email: neil.jassal@gmail.com

Updated 3/4232017

Implementation of Convolutional Neural Network
Adapted from:
https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet
"""

import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

keep_rate = 0.8

x = tf.placeholder('float')
y = tf.placeholder('float')


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
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

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


if __name__ == "__main__"():
    # load data
