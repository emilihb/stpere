from scipy import interpolate
import numpy as np


def get_sigma(covariance):
    """Returns sigma values from covariances.

    :param covariance: array of covariance matrices
    :type covariance: 3d array
    :return: array of sigma values
    :rtype: 1d array
    """
    sigma = np.empty(covariance.shape[0:2])
    for i in range(covariance.shape[1]):
        sigma[:, i] = covariance[:, i, i]
    sigma = np.sqrt(sigma)

    return sigma


def get_error(t1, v1, t2, v2):
    """Error respect to v2 interpolating values when
    timestamps do not match.

    :param t1: timestamps
    :param v1: value associated with t1
    :param t2: timestamps
    :param v2: value associated with t2
    :type t1: list or 1d array
    :type v1: 1d or 2d array
    :type t2: list or 1d array
    :type v2: 1d or 2d array (as v1)
    :return: error (v1 - v2')
    :rtype: list of floats
    """

    f = interpolate.interp1d(
        t2,
        v2,
        kind='linear',
        bounds_error=False,
        copy=False)
    new_v2 = f(t1)
    error = v1 - new_v2
    return error


def plot_error(x, y, sigma, xlabel, ylabel, title=''):
    """Simple error plot vs sigma and 3sigma. x and xlabel are 1d,
    y and ylabel are nd.
    """

    import matplotlib.pyplot as plt

    try:
        sigma3 = 3 * sigma

        fig = plt.figure(0)
        plt.title(title)

        if len(y.shape) > 1:
            idx = y.shape[1] * 100 + 11
            for i in range(y.shape[1]):
                plt.subplot(idx)
                plt.plot(x, sigma3[:, i], 'k', x, -sigma3[:, i], 'k')
                plt.plot(x, sigma[:, i], 'k--', x, -sigma[:, i], 'k--')
                plt.plot(x, y[:, i])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel[i])
                plt.tight_layout()
                idx += 1
        else:
            plt.subplot(111)
            plt.plot(x, sigma3, 'k', x, -sigma3, 'k')
            plt.plot(x, sigma, 'k--', x, -sigma, 'k--')
            plt.plot(x, y)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()

        return fig
    except Exception as e:
        print(e)
