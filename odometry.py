import numpy as np
from util import o2ca2_dataset as dataset
from util import measurement_8state as meas
from ekf import EKF8State, EKFBase
import time
import matplotlib.pyplot as plt


dvl_config = {
    "id": "sontek",
    "pose": np.array([0, 0, 0, np.pi, 0, np.pi/3]),
    "stdev_bottom": np.array([0.3, 0.3, 0.15]),  # m/s
    "stdev_water": np.array([0.6, 0.6, 0.3]),  # m/s
    "stdev_yaw": 0.2,  # rad
    "stdev_depth": 0.02  # m
    }

imu_config = {
    "id": "xsense",
    "pose": np.array([0, 0, -0.04, 0, 0, -np.pi/2]),
    "stdev_orientation": np.array([np.nan, np.nan, 0.2]),  # rad
    "stdev_angular_velocity": np.array([np.nan, np.nan, 0.2])  # rad/s
    }

mis_config = {
    "id": "micron",
    "pose": np.array([0.33, 0, -0.26, 0, 0, np.pi]),
    "stdev_linear": 0.1,
    "stdev_angular": 3*np.pi/180,
    "resolution_linear": 0.1,  # m
    "segmentation": {
        "threshold": 80,  # [0-255]
        "init_range": 1.5,  # m
        "max_range": 50,  # m
        "min_distance": 50,  # m
        "max_ranges": 2
        }
    }

ekf_config = {
    "stdev_velocity_model":  np.array([0.1, 0.05, 0.1, np.nan, np.nan, 0.25])  # m/s|rad/s
    }

filenames = {
    "dvl": "experiment3/_040825_1735_DVL.log",
    # "mis": "experiment3/_040825_1735_IS.log",
    # "gps": "experiment3/_040825_1735_DGPS.log",
    # "imu": "experiment3/_040825_1735_MTI.log"
    }



def update_time(curr_time, timestamp):
    """Compte delta time between 2 iterations"""
    prev_time = curr_time
    return (timestamp, timestamp-prev_time)

def main():
    print("Loading data..."),
    start = time.time()
    D = dataset.O2CA2Dataset(filenames,20)
    stop = time.time()
    print("[DONE] %.3fs" % (stop - start))

    gps_initialized = False
    update_trajectory = False
    timestamps = []
    odometry = []
    odometry_cov = []
    
    timestamps_imu = []
    imu = []
    timestamps_gps = []
    gps = []

    mis_timestamps = []
    mis_state = []
    mis_cov = []

    state = np.zeros(8)
    covariance = np.diag(np.ones(8))
    t = 0

    stdev_dvl = np.append(dvl_config["stdev_bottom"], dvl_config["stdev_depth"])
    stdev_imu = np.array([imu_config["stdev_orientation"][2], imu_config["stdev_angular_velocity"][2]])

    Q = np.diag(ekf_config["stdev_velocity_model"][[0, 1, 2, 5],]**2)
    ekf = EKF8State(Q)
    # try:
    #     while True:
    #         _type, data = D.next()
    #         if _type is "dvl":
    #             print data
    #             t, dt = update_time(t, data[0])
    #             z, H, R = meas.dvl(data, dvl_config["pose"], stdev_dvl)

    #             if not ekf. initialized:
    #                 # print "DVL initializes"
    #                 state = np.dot(H.T, z)
    #                 covariance = np.dot(np.dot(H.T, R), H)
    #                 ekf.set_initialized(True)
    #                 # print state, covariance
    #             else:
    #                 state, covariance = ekf.prediction(state, covariance, dt)
    #                 state, covariance = ekf.correction(z, H, R)
    #                 update_trajectory = True

    #         #     # print "z", z
    #         #     # print "H", H
    #         #     # print "R", R
    #         elif _type is "imu":
    #             t, dt = update_time(t, data[0])
    #             z, H, R = meas.imu(data, imu_config["pose"], stdev_imu)

    #             if not ekf._initialized:
    #                 # print "IMU initializes"
    #                 state = np.dot(H.T, z)
    #                 covariance = np.dot(np.dot(H.T, R), H)
    #                 ekf.set_initialized(True)
    #                 # print state, covariance
    #             else:
    #                 state, covariance = ekf.prediction(state, covariance, dt)
    #                 state, covariance = ekf.correction(z, H, R)

    #             timestamps_imu.append(t)
    #             imu.append(z[0])



    #         #     print "imu"
    #         #     # time_prev = time_curr
    #         #     # time_curr = data[0]
    #         #     # time_delta = time_curr - time_prev
    #         #     
    #         #     print "z", z
    #         #     print "H", H
    #         #     print "R", R


    #         #     # if not initialized:
    #         #     #     # print "IMU initializes"
    #         #     #     state = np.dot(H.T, z)
    #         #     #     covariance = np.dot(np.dot(H.T, R), H)
    #         #     #     initialized = True
    #         #     #     # print state, covariance
    #         #     # else:
    #         #     #     state, covariance = ekf.prediction(state, covariance, time_delta)
    #         #     #     state, covariance = ekf.correction(z, H, R)

    #         #     # timestamps_imu.append(time_curr)
    #         #     # imu.append([z[0, 0]])

    #         # elif _type is "mis":
    #         #     if initialized:
    #         #         time_prev = time_curr
    #         #         time_curr = data[0]
    #         #         time_delta = time_curr - time_prev

    #         #         state, covariance = ekf.prediction(state, covariance, time_delta)

    #         #         #temporary to generate a subdataset
    #         #         mis_timestamps.append(time_curr)
    #         #         mis_state.append(state)
    #         #         mis_cov.append(covariance)
    #         #         # tf = np.array([tf_mis_to_base[0], tf_mis_to_base[1], tf_mis_to_base[5]])
    #         #         # x = np.array([state[0], state[1], state[3]])
    #         #         # P = np.zeros((3, 3))
    #         #         # P[0:2, 0:2] = covariance[0:2, 0:2]
    #         #         # P[2, 2] = covariance[2, 2]
    #         #         # scan, cov = mis_scan(data, tf, mis_stdev, x, P)
    #         #         # print scan, cov
    #         #         # time.sleep(0.5)

    #         elif _type is "gps":
    #             utm = meas.gps(data)
    #             if not gps_initialized:
    #                 utm_init = utm - state[0:2]
    #                 gps_initialized = True

    #             timestamps_gps.append(data[0])
    #             gps.append(list(utm - utm_init))
            
    # except EOFError:
    #     print "DONE"

    # # gps= np.asarray(gps)
    # # fig2 = plt.figure(2)
    # # plt.title("Trajectory")
    # # plt.xlabel('X (m)')
    # # plt.ylabel('Y (m)')
    # # plt.axis('equal')
    # # # plt.plot(odometry[:, 0], odometry[:, 1], 'b')
    # # plt.plot(gps[:, 1], gps[:, 0], 'r+--') # plotting in local coords -> this should be fixed
    # # plt.tight_layout()
    # # fig2.show()
    # # plt.show()

if __name__ == "__main__":
    main()

