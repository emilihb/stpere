import numpy as np
from util import normalize

def compose(a, b, Pa=None, Pb=None):
    """2d composition between 'a' and 'b' with covariance -if provided.

    :param a: [x, y, theta] row/column array
    :param b: [x, y, theta] or [x, y] row/column array
    :param Pa: a covariance array
    :param Pb: b covariance array
    :returns composition with covariance
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]
    if len(b.shape) is 2:
        b = b.T[0]

    st = np.sin(a[2])
    ct = np.cos(a[2])

    P = None
    if len(a) is 3:
        if len(b) is 2:
            r = np.array([
                b[0]*ct - b[1]*st + a[0],
                b[0]*st + b[1]*ct + a[1]])

            if Pa is not None and Pb is not None:
                J1 = np.array([
                    [1, 0, -b[0]*st - b[1]*ct],
                    [0, 1, b[0]*ct - b[1]*st]])

                J2 = np.array([
                    [ct, -st],
                    [st, ct]])

                P = np.dot(np.dot(J1, Pa), J1.T) + np.dot(np.dot(J2, Pb), J2.T)
            return (r, P)

        elif len(b) is 3:
            r = np.array([
                b[0]*ct - b[1]*st + a[0],
                b[0]*st + b[1]*ct + a[1],
                normalize(a[2] + b[2])])

            if Pa is not None and Pb is not None:
                J1 = np.array([
                    [1, 0, -b[0]*st - b[1]*ct],
                    [0, 1, b[0]*ct - b[1]*st],
                    [0, 0, 1]])

                J2 = np.array([
                    [ct, -st, 0],
                    [st, ct, 0],
                    [0, 0, 1]])

                P = np.dot(np.dot(J1, Pa), J1.T) + np.dot(np.dot(J2, Pb), J2.T)
            return (r, P)


def inv(a, Pa=None):
    """2d inversion with covariance -if provided.

    :param a: [x, y, theta] row/column array
    :param Pa: a covariance array
    :returns: inversion with covariance
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]

    st = np.sin(a[2])
    ct = np.cos(a[2])

    P = None
    if len(a) is 3:
        r = np.array([
            -a[0]*ct - a[1]*st,
            a[0]*st - a[1]*ct,
            -a[2]])

        if Pa is not None:
            J = np.array([
                [-ct, -st, a[0]*st - a[1]*ct],
                [st, -ct, a[0]*ct + a[1]*st],
                [0, 0, -1]])

            P = np.dot(np.dot(J, Pa), J.T)
        return (r, P)


def main():
    a = np.array([5.0, 5.0, np.pi])
    Pa = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    b = np.array([1.0, 2.0, -np.pi])

    print "Compose:\n", compose(a, b)
    rinv,Pinv = inv(a, Pa)
    print "Inversion:\n", rinv
    print "Inversion:\n", compose(rinv, a)

if __name__ == "__main__":
    main()
