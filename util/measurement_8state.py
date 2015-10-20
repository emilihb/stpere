# from stpere.util import util
import util
import numpy as np


# def imu(msg, pose, stdev):
#     """Returns imu z, H, R formated for 4 degrees of freedom EKF.
#     z -- measurement
#     H -- observation model
#     R -- measurement covariance
#     """

#     # align imu axis to robot axis
#     orientation = util.normalize(np.array([-msg[2], msg[1], msg[3]-90.]) * np.pi/180)
#     # rot z -90 according to descrition
#     ang_velocity = np.array([msg[5], -msg[4], msg[6]])
#     lin_acceleration = np.array([msg[8], -msg[7], msg[9]])

#     z = np.array([orientation[2], ang_velocity[2]])
#     H = np.array([
#         [0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
#     R = np.diag(stdev**2)

#     return (z, H, R)


def dvl(msg, pose, stdev):
    """Returns dvl z, H, R formated for 4 degrees of freedom EKF.
    z -- measurement
    H -- observation model
    R -- measurement covariance
    """

    # bottom velocity
    if msg[14] == 1:
        rpy = util.rpy(pose[3], pose[4], pose[5])
        vel_dvl = msg[11:14] / 100.

    vel_base = np.dot(rpy, np.c_[vel_dvl]).T[0]
    depth = 0.003772250*(msg[26]-1440)

    z = np.r_[vel_base, depth]
    H = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0]], dtype=float)
    R = np.diag(stdev**2)

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



