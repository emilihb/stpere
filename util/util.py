import numpy as np


def normalize(angle):
    """Returns angle (in rads) in (-pi, pi] range ."""
    return angle + (2 * np.pi) * np.floor((np.pi - angle) / (2 * np.pi))


def rpy(roll, pitch, yaw):
    """Returns 3d rotation matrix in rads."""
    sr = np.sin(roll)
    cr = np.cos(roll)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sy = np.sin(yaw)
    cy = np.cos(yaw)

    return np.array([
        [cy*cp, -sy*cr + cy*sp*sr, sy*sr + cy*sp*cr],
        [sy*cp, cy*cr + sr*sy*sp, -sr*cy + sy*sp*cr],
        [-sp, cp*sr, cp*cr]])



# def homogeneous_matrix(x, y, z, roll, pitch, yaw):
#     """Returns 3d homogeneous matrix (4x4)."""
#     sr = np.sin(roll)
#     cr = np.cos(roll)
#     sp = np.sin(pitch)
#     cp = np.cos(pitch)
#     sy = np.sin(yaw)
#     cy = np.cos(yaw)

#     R = np.array([[cy*cp, -sy*cr + cy*sp*sr, sy*sr + cy*sp*cr, x],
#                   [sy*cp, cy*cr + sr*sy*sp, -sr*cy + sy*sp*cr, y],
#                   [-sp, cp*sr, cp*cr, z],
#                   [0, 0, 0, 1]])
#     return R


# def homogeneous_matrix_2d(x, y, theta):
#     """Returns 3d homogeneous matrix (3x3)."""
#     st = np.sin(theta)
#     ct = np.cos(theta)

#     R = np.array([[ct, -st, x], [st, ct, y], [0, 0, 1]])
#     return R


# def pol_to_cart(rho, phi):
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return np.asarray((x, y)).T


# def main():
#     # print "Normalize: ", normalize(3 * np.pi)
#     # print "RPY: ", rotation_matrix(0, 0, np.pi)
#     # print "Homogeneous matrix:", homogeneous_matrix(2, 4, 6, 0, 0, np.pi)
#     # print "Homogeneous matrix 2d:", homogeneous_matrix_2d(2, 4, np.pi)
#     print "Compounding:\n", compose(np.array([2, 2, np.pi]), np.array([1, 4, 0.5]))
#     print "Inversion:\n", inv(np.array([2, 2, np.pi]))


# if __name__ == "__main__":
#     main()
