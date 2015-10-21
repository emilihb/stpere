import numpy as np
from util import o2ca2_dataset as dataset
from util import measurement_8state as meas
from ekf.ekf import EKF8State
from ekf.tools import *
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
    "pose": np.array([0, 0, -0.04, 0, 0, -86*np.pi/180]),
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
    "mis": "experiment3/_040825_1735_IS_timestamps.log",
    "gps": "experiment3/_040825_1735_DGPS.log",
    "imu": "experiment3/_040825_1735_MTI.log"
    }


def update_time(curr_time, timestamp):
    """Compte delta time between 2 iterations"""
    prev_time = curr_time
    return (timestamp, timestamp-prev_time)


def main():
    print("Loading data..."),
    start = time.time()
    D = dataset.O2CA2Dataset(filenames)
    stop = time.time()
    print("[DONE]  %.3fs" % (stop-start))

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

    stdev_dvl = {
        "bottom": np.append(dvl_config["stdev_bottom"], dvl_config["stdev_depth"]),
        "water": np.append(dvl_config["stdev_water"], dvl_config["stdev_depth"])
        }

    stdev_imu = np.array([imu_config["stdev_orientation"][2], imu_config["stdev_angular_velocity"][2]])

    Q = np.diag(ekf_config["stdev_velocity_model"][[0, 1, 2, 5], ]**2)
    ekf = EKF8State(Q)
    try:
        while True:
            _type, data = D.next()
            if _type is "dvl":
                t, dt = update_time(t, data[0])
                z, H, R = meas.dvl(data, dvl_config["pose"], stdev_dvl)

                if not ekf._initialized:
                    state = np.dot(H.T, z)
                    covariance = np.dot(np.dot(H.T, R), H)
                    ekf.set_initialized(True)
                else:
                    state, covariance = ekf.prediction(state, covariance, dt)
                    state, covariance = ekf.correction(z, H, R)
                    update_trajectory = True

            elif _type is "imu":
                t, dt = update_time(t, data[0])
                z, H, R = meas.imu(data, imu_config["pose"], stdev_imu)

                if not ekf._initialized:
                    state = np.dot(H.T, z)
                    covariance = np.dot(np.dot(H.T, R), H)
                    ekf.set_initialized(True)
                else:
                    state, covariance = ekf.prediction(state, covariance, dt)
                    state, covariance = ekf.correction(z, H, R)

                timestamps_imu.append(t)
                imu.append(z[0])

            elif _type is "mis":
                if ekf._initialized:
                    t, dt = update_time(t, data[0])
                    state, covariance = ekf.prediction(state, covariance, dt)

                    # temporary to generate a subdataset
                    mis_timestamps.append(t)
                    mis_state.append(state)
                    mis_cov.append(covariance)

            elif _type is "gps":
                utm = meas.gps(data)
                if not gps_initialized:
                    utm_init = utm - state[0:2]
                    gps_initialized = True

                timestamps_gps.append(data[0])
                gps.append(list(utm - utm_init))

            if update_trajectory:
                timestamps.append(t)
                odometry.append(state[[0, 1, 2, 3], ])
                odometry_cov.append(covariance)
                update_trajectory = False
    except EOFError:
        pass

    timestamps = np.asarray(timestamps)
    odometry = np.asarray(odometry)
    odometry_cov = np.asarray(odometry_cov)
    timestamps_gps = np.asarray(timestamps_gps)
    gps = np.asarray(gps)
    timestamps_imu = np.asarray(timestamps_imu)
    imu = np.asarray(imu)

    print("Computing error..."),
    start = time.time()
    err = get_error(timestamps, odometry[:, [0, 1]].T, timestamps_gps, gps[:, ::-1].T).T
    err_yaw = get_error(timestamps, odometry[:, 3], timestamps_imu, imu.T).T
    stop = time.time()
    print("[DONE] %.3fs" % (stop - start))
    sigma = get_sigma(odometry_cov)

    timestamps = timestamps - timestamps[0]
    fig1 = plot_error(
        x=timestamps,
        y=np.c_[err[:, 0:2], err_yaw],
        sigma=sigma[:, [0, 1, 3]],
        xlabel='time [s]',
        ylabel=['X [m]', 'Y [m]', '$\psi$ [rad]'],
        title="Errors")
    fig1.show()

    fig2 = plt.figure(2)
    plt.title("Trajectory")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.plot(odometry[:, 0], odometry[:, 1], 'b')
    plt.plot(gps[:, 1], gps[:, 0], 'r+--')  # plotting in local coords -> this should be fixed
    plt.tight_layout()
    fig2.show()

    plt.show()

if __name__ == "__main__":
    main()

