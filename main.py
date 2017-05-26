#!/usr/bin/python3
"""
Author: Jiaqing Lin
E-mail: jiaqing930@gmail.com
The source code of Backpropagation(BP) neural network.
"""

import numpy as np
import struct
import time

def read_data(datatype='train'):
    """
    Read MNIST dataset.
    Get training dataset if set datatype='train'.
    Get testing dataset if set datatype='test'.
    """
    # Set each file path.
    if datatype is 'train':
        image_file = "./dataset/train-images-idx3-ubyte"
        label_file = "./dataset/train-labels-idx1-ubyte"
    if datatype is 'test':
        image_file = "./dataset/t10k-images-idx3-ubyte"
        label_file = "./dataset/t10k-labels-idx1-ubyte"

    if image_file and label_file:
        # Read labels data from given file.
        with open(label_file, 'rb') as label_f:
            magic, num_item = struct.unpack('>II', label_f.read(8))
            label_tmp = np.fromfile(label_f, dtype=np.uint8)
            labels = label_tmp.reshape((np.int32(num_item), np.int32(1)))
        # Read images data from given file.
        with open(image_file, 'rb') as image_f:
            magic, num_item, rows, cols = struct.unpack('>IIII', image_f.read(16))
            img_tmp = np.fromfile(image_f, dtype=np.uint8)
            images = img_tmp.reshape((np.int32(num_item), np.int32(rows * cols))) / 255.0
        images = np.asarray(images)
        labels = np.asarray(labels)
        return images, labels
    else:
        print("datatype must be 'train' or 'test'.\n")


def sigmoid(x, derive=False):
    """
    Create Sigmoid function.
    """
    if derive is True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def training(images_train, labels_train):
    """
    Using dataset to train BP neural network.
    """
    print("Training...")
    start_t = time.time()
    data_num, inChannel = images_train.shape
    error = []
    # Number of neurons in hidden layer.
    hidden_num = 40
    # Number of train target.
    outChannel_num = 10
    learning_rate = 0.2
    np.random.seed(1)
    w1 = 0.2 * np.random.random((inChannel, hidden_num)) - 0.1
    w2 = 0.2 * np.random.random((hidden_num, outChannel_num)) - 0.1
    hidden_offset = np.zeros(hidden_num)
    outChannel_offset = np.zeros(outChannel_num)

    for i in range(0, data_num):
        label = np.zeros(outChannel_num)
        label[labels_train[i]] = 1
        # Set 3 layers network.
        l0 = images_train[i]
        l1 = sigmoid(np.dot(l0, w1) + hidden_offset)
        l2 = sigmoid(np.dot(l1, w2) + outChannel_offset)
        l2_error = label - l2
        error.append(np.mean(np.abs(l2_error)))
        # Link each layers.
        l2_delta = l2_error * sigmoid(l2, derive=True)
        l1_error = l2_delta.dot(w2.T)
        l1_delta = l1_error * sigmoid(l1, derive=True)
        for j in range(0, outChannel_num):
            w2[ : , j] += learning_rate * l2_delta[j] * l1
        for k in range(0, hidden_num):
            w1[ : , k] += learning_rate * l1_delta[k] * l0

        outChannel_offset += learning_rate * l2_delta
        hidden_offset += learning_rate * l1_delta
    print("error: {0:.3g}       times: {1:.4g} seconds".format(sum(error) / data_num, time.time() - start_t))
    return w1, w2, hidden_offset, outChannel_offset


def testing(images_test, labels_test, w1, w2, hidden_offset, outChannel_offset):
    """
    Using dataset to test trained BP neural network.
    """
    print("Testing...")
    start_t = time.time()
    data_num, inChannel = images_test.shape
    right = np.zeros(10)
    for i in range(0, data_num):
        l0 = images_test[i]
        l1 = sigmoid(np.dot(l0, w1) + hidden_offset)
        l2 = sigmoid(np.dot(l1, w2) + outChannel_offset)
        if np.argmax(l2) == labels_test[i]:
            right[labels_test[i]] += 1
    print("accuracy: {0}    times: {1:.4g} seconds".format(right.sum() / data_num, time.time() - start_t))


if __name__ == "__main__":
    images_train, labels_train = read_data(datatype='train')
    images_test, labels_test = read_data(datatype='test')
    # Training.
    w1, w2, hidden_offset, outChannel_offset = training(images_train, labels_train)
    # Testing.
    testing(images_test, labels_test, w1, w2, hidden_offset, outChannel_offset)
