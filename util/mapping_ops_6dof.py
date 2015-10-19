import numpy as np
from util import normalize

def compose(a, b, Pa=None, Pb=None):
    """6DoF composition between 'a' and 'b' with covariance -if provided.

    :param a: [x, y, z, roll, pitch, yaw] row/column vector
    :param b: [x, y, z, roll, pitch, yaw] or [x, y, z] row/column vector
    :param Pa: a 6x6 covariance array
    :param Pb: b 6x6 or 3x3 covariance array
    :type a: 6-element array
    :type b: 3 or 6-element array
    :type Pa: 6x6 array
    :type Pb: 3x3 or 6x6 array
    :return: composition with covariance -if provided
    :rtype: 2-element tuple (r, P) or (r, None)
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]
    if len(b.shape) is 2:
        b = b.T[0]

    sr = np.sin(a[3])
    cr = np.cos(a[3])
    sp = np.sin(a[4])
    cp = np.cos(a[4])
    sy = np.sin(a[5])
    cy = np.cos(a[5])

    r = np.array([
        a[0] + b[0]*cp*cy + b[1]*(sp*sr*cy - sy*cr) + b[2]*(sp*cr*cy + sr*sy),
        b[0]*sy*cp + a[1] + b[1]*(sp*sr*sy + cr*cy) + b[2]*(sp*sy*cr - sr*cy),
        -b[0]*sp + b[1]*sr*cp + a[2] + b[2]*cp*cr])

    if len(b) is 6:
        r = np.r_[r, normalize(a[3:] + b[3:])]

    P = None
    if Pa is not None and Pb is not None:
        Ja = np.array([
            [1, 0, 0, b[1]*(sp*cr*cy + sr*sy) + b[2]*(-sp*sr*cy + sy*cr), -b[0]*sp*cy + b[1]*sr*cp*cy + b[2]*cp*cr*cy, -b[0]*sy*cp + b[1]*(-sp*sr*sy - cr*cy) + b[2]*(-sp*sy*cr + sr*cy)],
            [0, 1, 0, b[1]*(sp*sy*cr - sr*cy) + b[2]*(-sp*sr*sy - cr*cy), -b[0]*sp*sy + b[1]*sr*sy*cp + b[2]*sy*cp*cr, b[0]*cp*cy + b[1]*(sp*sr*cy - sy*cr) + b[2]*(sp*cr*cy + sr*sy)],
            [0, 0, 1, b[1]*cp*cr - b[2]*sr*cp, -b[0]*cp - b[1]*sp*sr - b[2]*sp*cr, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])

        Jb = np.array([
            [cp*cy, sp*sr*cy - sy*cr, sp*cr*cy + sr*sy, 0, 0, 0],
            [sy*cp, sp*sr*sy + cr*cy, sp*sy*cr - sr*cy, 0, 0, 0],
            [-sp, sr*cp, cp*cr, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        if len(Pb) is 3:
            P = np.dot(np.dot(Ja[0:3], Pa), Ja[0:3].T) + np.dot(np.dot(Jb[0:3, 0:3], Pb), Jb[0:3, 0:3].T)
        else:
            P = np.dot(np.dot(Ja, Pa), Ja.T) + np.dot(np.dot(Jb, Pb), Jb.T)

    return (r, P)


def inv(a, Pa=None):
    """6DoF inversion with covariance -if provided.

    :param a: [x, y, z, roll, pitch, yaw] row/column vector
    :param Pa: a 6x6 covariance array
    :type a: 6-element array
    :return: inversion with covariance -if provided
    :rtype: 2-element tuple (r, P) or (r, None);
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]

    sr = np.sin(a[3])
    cr = np.cos(a[3])
    sp = np.sin(a[4])
    cp = np.cos(a[4])
    sy = np.sin(a[5])
    cy = np.cos(a[5])

    r = np.array([
        -a[0]*cp*cy - a[1]*(sp*sr*cy + sy*cr) - a[2]*(-sp*cr*cy + sr*sy),
        a[0]*sy*cp - a[1]*(-sp*sr*sy + cr*cy) - a[2]*(sp*sy*cr + sr*cy),
        -a[0]*sp + a[1]*sr*cp - a[2]*cp*cr,
        -a[3],
        -a[4],
        -a[5]])

    P = None
    if Pa is not None:
        Ja = np.array([
            [-cp*cy, -sp*sr*cy - sy*cr,  sp*cr*cy - sr*sy,   -a[1]*(sp*cr*cy - sr*sy) - a[2]*(sp*sr*cy + sy*cr),  a[0]*sp*cy - a[1]*sr*cp*cy + a[2]*cp*cr*cy, a[0]*sy*cp - a[1]*(-sp*sr*sy + cr*cy) - a[2]*(sp*sy*cr + sr*cy)],
            [sy*cp,  sp*sr*sy - cr*cy, -sp*sy*cr - sr*cy, -a[1]*(-sp*sy*cr - sr*cy) - a[2]*(-sp*sr*sy + cr*cy), -a[0]*sp*sy + a[1]*sr*sy*cp - a[2]*sy*cp*cr, a[0]*cp*cy - a[1]*(-sp*sr*cy - sy*cr) - a[2]*(sp*cr*cy - sr*sy)],
            [-sp, sr*cp, -cp*cr, a[1]*cp*cr + a[2]*sr*cp, -a[0]*cp - a[1]*sp*sr + a[2]*sp*cr, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1]])
        P = np.dot(np.dot(Ja, Pa), Ja.T)
    return (r, P)


def main():
    a = np.array([2, 2, 2, np.pi, np.pi, np.pi])
    Pa = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    b = np.array([2, 2, 2])
    Pb = np.diag([0.1, 0.1, 0.1])
    b1 = np.array([2, 2, 2, np.pi, np.pi, np.pi])
    Pb1 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    print "Compose:\n", compose(a, b1)
    print "Compose:\n", compose(a, b1, Pa, Pb1)
    print "Compose:\n", compose(a, b, Pa, Pb)
    print "Inv:\n", inv(a, Pa)

if __name__ == "__main__":
    main()
