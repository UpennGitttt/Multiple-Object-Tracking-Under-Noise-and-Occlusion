import numpy as np


def kf(observation, state, filterstep, error_cov):
    A = np.array([[1, 0, filterstep * np.cos(state[3]), -filterstep * state[2] * np.sin(state[3])],
                  [0, 1, filterstep * np.sin(state[3]), filterstep * state[2] * np.cos(state[3])],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Dynamic/process noise
    R = np.diag([0.05, 0.05, 0.05, 0.05]) * 0.001
    # Observation noise
    Q = np.diag([.000001, .000001])

    # Propagate
    state = A @ state
    error_cov = A @ error_cov @ A.T + R

    # Incorporate observations
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    kalman_gain = error_cov @ H.T @ np.linalg.inv(H @ error_cov @ H.T + Q)
    state = state + (kalman_gain @ (observation[0:2] - H @ state)).flatten()
    error_cov = (np.eye(4) - kalman_gain @ H) @ error_cov

    return state, error_cov


def kf_3d(cur_state, obs, P, dt):
    # state = (x, y, z, dx, dy, dz, r, p, y, dr, dp, dy)
    A_1 = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],  # dynamic matrix
                     [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    A_2 = np.zeros((6, 6))
    A = np.vstack((np.hstack((A_1, A_2)), np.hstack((A_2, A_1))))
    Q = 0.05 * np.identity(12)
    H = np.identity(12)
    R = 0.0001 * np.identity(12)
    # Project the state ahead
    cur_state = A @ cur_state
    # Project the error covariance ahead
    P = A @ P @ A.T + Q
    # Measurement Update (Correction)
    # Compute the Kalman Gain
    S = H @ P @ H.T + R
    K = (P @ H.T) @ np.linalg.pinv(S)

    Z = obs.copy()
    # 计算残差
    y = (Z - cur_state).reshape(-1)
    # 更新
    cur_state = cur_state + np.matmul(K, y)
    # Update the error covariance
    P = np.matmul((np.identity(12) - np.matmul(K, H)), P)

    return cur_state, P

def kf_3d_missed(cur_state, obs, P, dt):
    # state = (x, y, z, dx, dy, dz, r, p, y, dr, dp, dy)
    A_1 = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],  # dynamic matrix
                     [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    A_2 = np.zeros((6, 6))
    A_3 = np.eye(6)
    A = np.vstack((np.hstack((A_1, A_2)), np.hstack((A_2, A_3))))
    Q = 0.05 * np.identity(12)
    H = np.identity(6)
    R = 0.0001 * np.identity(6)
    # Project the state ahead
    cur_state = A @ cur_state
    # Project the error covariance ahead
    P = A @ P @ A.T + Q
    # Measurement Update (Correction)
    # Compute the Kalman Gain
    S = H @ P[:6, :6] @ H.T + R
    K = (P[:6, :6] @ H.T) @ np.linalg.pinv(S)

    Z = obs.copy()
    # 计算残差
    y = Z - cur_state[:6]
    # 更新
    cur_state[:6] = cur_state[:6] + np.matmul(K, y)
    # Update the error covariance
    P[:6, :6] = np.matmul((np.identity(6) - np.matmul(K, H)), P[:6, :6])

    return cur_state, P

