import util
import numpy as np

""" 8-state [x, y ,z, yaw, u, v, w, r] measurements from O2CA2 datasets
    to be used in a EKF.
    Look at http://eia.udg.edu/~dribas/files/description.pdf for more
    information
"""


def imu(msg, pose, stdev):
    """Returns IMU measurement z, observation model H and
    measurement covariance R formatted for an 8-state EKF.

    :param msg: data msg from O2CA2 dataset
    :param pose: imu pose [x, y, z, roll, pitch, yaw] from the robot 0
    :param stdev: standard devidation of the measurements
    :type msg: 1d array
    :type pose: 1d array
    :type stdev: dictionary of type (key: 4-element array)
    :return: (z, H, R)
    :rtype: 3-element tuple
    """

    # align imu axis to robot axis
    orientation = util.normalize(np.array([-msg[2], msg[1], msg[3]])*np.pi/180 + pose[3:6])
    # rot z -90 according to descrition
    ang_velocity = np.array([msg[5], -msg[4], msg[6]])
    lin_acceleration = np.array([msg[8], -msg[7], msg[9]])

    # z = np.array([orientation[2], ang_velocity[2]])
    # H = np.array([
    #     [0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
    # R = np.diag(stdev**2)

    # lower error. still don't know why
    z = np.array([orientation[2]])
    H = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0]], dtype=float)
    R = np.diag([stdev[0]**2])
    return (z, H, R)


def dvl(msg, pose, stdev):
    """Returns DVL measurement z, observation model H and
    measurement covariance R formatted for an 8-state EKF.

    :param msg: data msg from O2CA2 dataset
    :param pose: dvl pose [x, y, z, roll, pitch, yaw] from the robot 0
    :param stdev: standard devidation of the measurements
    :type msg: 1d array
    :type pose: 1d array
    :type stdev: dictionary of type (key: 4-element array)
    :return: (z, H, R)
    :rtype: 3-element tuple
    """

    # bottom velocity
    if msg[14] == 1:
        rpy = util.rpy(pose[3], pose[4], pose[5])
        vel_dvl = msg[11:14] / 100.
        cov = stdev["bottom"]**2
    else:  # water velocity
        rot = msg[24:21:-1] * np.pi / 180
        rpy = util.rpy(rot[0], rot[1], rot[2])
        vel_dvl = msg[7:10] / 100
        cov = stdev["water"]**2

    vel_base = np.dot(rpy, np.c_[vel_dvl]).T[0]
    depth = 0.003772250*(msg[26]-1440)

    z = np.r_[vel_base, depth]
    H = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0]], dtype=float)
    R = np.diag(cov)

    return (z, H, R)


def gps(msg, proj='utm', zone='31T', datum='WGS84'):
    """Returns gps coordinates in Easings and Northings.

    :param msg: data msg from O2CA2 dataset
    :param proj: projection type
    :param zone: UTM zone
    :param datum: geodeic datum
    :return: (x, y) in UTM
    :rtype: 2-element tuple
    """

    from pyproj import Proj
    proj = Proj(proj=proj, zone=zone, datum=datum)
    latlon = np.floor(msg[1:3]/100)+((msg[1:3]/100)-np.floor(msg[1:3]/100))*100/60
    x, y = proj(latlon[1], latlon[0])
    return (x, y)
