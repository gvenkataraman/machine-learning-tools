"""
Simple neural network using tensorflow
NOTE: in this case, adding additional layer hurts!
So, this is just for practice and understanding tf
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def run():
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
    # the input variables
    # number of dimensions in the input
    n = 784
    # number of classes
    c = 10

    # input
    x = tf.placeholder(tf.float32, [None, n])
    y_ = tf.placeholder(tf.float32, [None, c])

    # weights to deduce in the model
    # arbitrary number of dimensions in hidden layer
    Dh = 784

    # to get hidden layer
    W1 = tf.Variable(tf.zeros([n, Dh]))
    b1 = tf.Variable(tf.zeros([Dh]))

    # output layer
    W2 = tf.Variable(tf.zeros([Dh, c]))
    b2 = tf.Variable(tf.zeros([c]))

    # hidden layer
    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    # response
    y = tf.nn.softmax(tf.matmul(h, W2) + b2)

    #loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    run()