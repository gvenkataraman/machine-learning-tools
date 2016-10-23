"""
Simple softmax using tensorflow
do not use built in function, but compute it from 'scratch'
"""
import numpy as np
import tensorflow as tf


def softmax_tf(x):
    x_max = tf.reduce_max(x, reduction_indices=[1])
    x_max = tf.expand_dims(x_max, dim=1)
    x_sub = x - x_max
    x_exp = tf.exp(x_sub)
    x_exp_sum = tf.reduce_sum(x_exp, reduction_indices=[1])
    x_exp_sum = tf.expand_dims(x_exp_sum, dim=1)
    return x_exp / x_exp_sum


def test_softmax_tf():
    softmax_input = tf.convert_to_tensor(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32)
    test1 = softmax_tf(softmax_input)
    with tf.Session():
        test1 = test1.eval()
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142, 0.73105858]))) <= 1e-6


if __name__ == '__main__':
    test_softmax_tf()