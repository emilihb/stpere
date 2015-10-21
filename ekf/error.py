from scipy import interpolate
import numpy as np

def get_sigma(cov):
    """Returns sigma values from covariances"""
    sigma = np.empty((cov.shape[0], cov.shape[1]))
    for i in range(cov.shape[1]):
        sigma[:, i] = cov[:, i, i]
    sigma = np.sqrt(sigma)

    return sigma


def get_error(t, state, time_gtruth, gtruth):
    """Error computation respect ground gruth interpolating values when
    timestamps dont match
    state is NxM, for N degrees of freedom
    gtruth is NxO for N degrees of freedom"""

    f = interpolate.interp1d(
        time_gtruth,
        gtruth,
        kind='linear',
        bounds_error=False,
        copy=False)
    new_gtruth = f(t)
    error = state - new_gtruth
    return error