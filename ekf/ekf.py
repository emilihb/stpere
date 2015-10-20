import numpy as np


class EKFBase(object):
    """
    Base class for EKFs
    """
    def __init__(self, Q, state_size):
        self._Q = Q  # Model Uncertainty
        self._x = []
        self._P = []
        self._I = np.eye(state_size)
        self._initialized = False

    def correction(self, z, H, R):
        """ EKF correction/update with a measurement z,
        observation model H, observation noise R

        :param z: measurement
        :param H: observation model
        :param R: observation noise
        :type z: TODO
        :type H: TODO
        :type R: TODO
        :return: updated state x and covariance P
        :rtype: 2-emement tuple (x, P)
        """
        x = np.c_[self._x]
        z = np.c_[z]

        h = np.dot(H, x)
        y = z - h
        S = np.dot(np.dot(H, self._P), H.T) + R
        K = np.dot(np.dot(self._P, H.T), np.linalg.inv(S))

        self._x = (x + np.dot(K, y)).T[0]  # 1d array
        self._P = np.dot((self._I - np.dot(K, H)), self._P)
        return (self._x, self._P)

    def set_initialized(self, value):
        self._initialized = value


class EKF8State(EKFBase):
    """
    4 dof EKF, 8-element in vector state: [x, y, z, yaw, u, v, w, r]
    with constant veloicity model and acceleration noise
    """
    def __init__(self, Q):
        super(EKF8State, self).__init__(Q, 8)

    def __constant_velocity_model(self, x_prev, dt):
        cy = np.cos(x_prev[3])
        sy = np.sin(x_prev[3])

        x = np.array([
            x_prev[0] + x_prev[4]*dt*cy - x_prev[5]*dt*sy,
            x_prev[1] + x_prev[4]*dt*sy + x_prev[5]*dt*cy,
            x_prev[2] + x_prev[6]*dt,
            x_prev[3] + x_prev[7]*dt,
            x_prev[4],
            x_prev[5],
            x_prev[6],
            x_prev[7]])

        # Jacobian respect to vector x
        Jx = np.array([
            [1, 0, 0, -(x_prev[4]*dt)*sy-(x_prev[5]*dt)*cy, dt*cy, -dt*sy, 0, 0],
            [0, 1, 0,  (x_prev[4]*dt)*cy-(x_prev[5]*dt)*sy, dt*sy,  dt*cy, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]])
        return (x, Jx)

    def __acceleration_noise_jacobian(self, x_prev, dt):
        cy = np.cos(x_prev[3])
        sy = np.sin(x_prev[3])
        dt2 = dt**2

        return np.array([
            [0.5*dt2*cy, -0.5*dt2*sy, 0, -0.5*(x_prev[4]*dt)*dt2*sy-0.5*(x_prev[5]*dt)*dt2*cy],
            [0.5*dt2*sy,  0.5*dt2*cy, 0,  0.5*(x_prev[4]*dt)*dt2*cy-0.5*(x_prev[5]*dt)*dt2*sy],
            [0, 0, 0.5*dt2, 0],
            [0, 0, 0, 0.5*dt2],
            [dt, 0, 0, 0],
            [0, dt, 0, 0],
            [0, 0, dt, 0],
            [0, 0, 0, dt]])

    def prediction(self, x_prev, P_prev, dt):
    	""" EKF prediction based on the previous state x_prev and
        covariance P_prev using a constant velocity model.

        :param x_prev: vector state at t-1 [x, y, z, yaw, u, v, w, r]
        :param P_prev: covariance at t-1
        :param dt: delta time abs(t - t-1)
        :type x_prev: 1x8 array
        :type P_prev: 8x8 array
        :type dt: float
        :return: predition x and covariance P
        :rtype: 2-emement tuple (x, P)
        """
        if len(x_prev.shape) is 2:
            x_prev = x_prev.T[0]

        self._x, Jx = self.__constant_velocity_model(x_prev, dt)
        Jq = self.__acceleration_noise_jacobian(x_prev, dt)

        self._P = np.dot(np.dot(Jx, P_prev), Jx.T) + np.dot(np.dot(Jq, self._Q), Jq.T)
        return (self._x, self._P)


if __name__ == "__main__":
    Q = np.diag([0.2**2, 0.2**2, 0.2**2, 0.05**2])
    x = np.zeros(8)+0.1
    P = np.diag(np.ones(8) * 1000)

    z = np.array([0.24, -0.05, -0.01, 0.18])
    H = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0]])
    R = np.array([
        [0.1, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 0.1, 0],
        [0, 0, 0, 0.1]])

    ekf = EKF8State(Q)
    p = ekf.prediction(x, P, 0.5)
    print ("Prediction", p)
    c = ekf.correction(z, H, R)
    print ("Correction", c)
