import numpy as np
from util import normalize


def compose(a, b, Pa=None, Pb=None):
    """3dof [x, y, theta] composition between 'a' and 'b'
    with covariance -if provided.

    :param a: [x, y, theta] row/column vector
    :param b: [x, y, theta] or [x, y] row/column vector
    :param Pa: a covariance array
    :param Pb: b covariance array
    :type a: 3-element array
    :type b: 2 or 3-element array
    :type Pa: 3x3 array
    :type Pb: 2x2 or 3x3 array
    :return: composition with covariance -if provided
    :rtype: 2-element tuple (r, P) or (r, None)
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]
    if len(b.shape) is 2:
        b = b.T[0]

    st = np.sin(a[2])
    ct = np.cos(a[2])

    r = np.array([
        b[0]*ct - b[1]*st + a[0],
        b[0]*st + b[1]*ct + a[1]])

    if len(b) is 3:
        r = np.r_[r, normalize(a[2:] + b[2:])]

    P = None
    if Pa is not None and Pb is not None:
        J1 = np.array([
            [1, 0, -b[0]*st - b[1]*ct],
            [0, 1, b[0]*ct - b[1]*st],
            [0, 0, 1]])

        J2 = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]])
        if len(Pb) is 2:
            P = np.dot(np.dot(J1[0:2], Pa), J1[0:2].T) + np.dot(np.dot(J2[0:2, 0:2], Pb), J2[0:2, 0:2].T)
        else:
            P = np.dot(np.dot(J1, Pa), J1.T) + np.dot(np.dot(J2, Pb), J2.T)
    return (r, P)


def inv(a, Pa=None):
    """3dof [x, y, theta] inversion with covariance -if provided.

    :param a: [x, y, theta] row/column vector
    :param Pa: a covariance vector
    :type a: 3-element array
    :type Pa: 3x3 array
    :return: inversion with covariance -if provided
    :rtype: 2-element tuple (r, P) or (r, None)
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]

    st = np.sin(a[2])
    ct = np.cos(a[2])

    r = np.array([
        -a[0]*ct - a[1]*st,
        a[0]*st - a[1]*ct,
        -a[2]])

    P = None
    if Pa is not None:
        Ja = np.array([
            [-ct, -st, a[0]*st - a[1]*ct],
            [st, -ct, a[0]*ct + a[1]*st],
            [0, 0, -1]])

        P = np.dot(np.dot(Ja, Pa), Ja.T)
    return (r, P)


def main():
    a = np.array([2.0, 2.0, np.pi])
    Pa = np.diag([0.1, 0.1, 0.1])
    b = np.array([2.0, 2.0])
    Pb = np.diag([0.1, 0.1])
    b1 = np.array([2.0, 2.0, np.pi])
    Pb1 = np.diag([0.1, 0.1, 0.1])

    print "Compose:\n", compose(a, b1)
    print "Compose:\n", compose(a, b1, Pa, Pb1)
    print "Compose:\n", compose(a, b, Pa, Pb)
    print "Inv:\n", inv(a, Pa)

if __name__ == "__main__":
    main()
