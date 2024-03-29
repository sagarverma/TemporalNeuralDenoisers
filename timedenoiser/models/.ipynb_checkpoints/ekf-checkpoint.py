import numpy as np
from pykalman import KalmanFilter


def ekf(noisy, std):
    transition_covariance = np.diag([std, 0]) ** 2
    transition_matrix = np.asarray([[1, 0], [0, 1]])
    kf = KalmanFilter(transition_matrices=transition_matrix,
                      transition_covariance=transition_covariance)
    clean_kf, state_cov = kf.smooth(noisy)

    return clean_kf[:, 0]
