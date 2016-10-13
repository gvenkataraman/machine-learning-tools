import numpy as np


def softmax_numpy(x):
    """
    :param x:  numpy array of shape (n_samples, n_features)
    :return: numpy array of shape (n_samples, n_features)
    Things that are important in this code:
    1. delegate looping into numpy and don't use for loops in python
    2. Use broadcasting to to operations on vectors/matrices with different shapes
    a good article on broadcasting can be found here:
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    3. Direct computation of exp(x) can lead to overflow, use the property:
        softmax(x) = softmax(x+c) for any constant c
    """
    x_max = np.max(x, axis=1)
    x_exp = np.exp((x.transpose() - x_max).transpose())
    x_exp_sum = np.sum(x_exp, axis=1)
    return x_exp / x_exp_sum[:, None]


def test_softmax_numpy():
    """
     Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax_numpy(np.array([[1001, 1002], [3, 4]]))
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142, 0.73105858]))) <= 1e-6
    test2 = softmax_numpy(np.array([[-1001, -1002]]))
    assert np.amax(np.fabs(test2 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6
    print "softmax tests pass\n"


if __name__ == "__main__":
    test_softmax_numpy()
