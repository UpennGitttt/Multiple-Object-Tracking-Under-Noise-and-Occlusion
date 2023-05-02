import numpy as np


def ekf(m, x, filterstep, error_cov):
    I = np.eye(4)

    # Project the state ahead
    # g(x)的解析式
    x[0] = x[0] + filterstep * x[3] * np.cos(x[2])
    x[1] = x[1] + filterstep * x[3] * np.sin(x[2])
    # 保持角度在[-np.pi, np.pi]之间
    x[2] = (x[2] + np.pi) % (2.0 * np.pi) - np.pi  # 角度
    x[3] = x[3]  # 速度

    # Calculate the Jacobian of the Dynamic Matrix A
    # 线性化卡尔曼矩阵
    # g对x求导得到的
    a13 = filterstep * x[3] * np.sin(x[2])
    a14 = filterstep * np.cos(x[2])
    a23 = filterstep * x[3] * np.cos(x[2])
    a24 = filterstep * np.sin(x[2])
    JA = np.array([[1.0, 0.0, a13, a14],
                   [0.0, 1.0, a23, a24],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # Process noise cov
    Q = np.diag([0.05, 0.05, 0.05, 0.05]) * 0.001

    # Project the error covariance ahead
    # 更新噪声 J_A @ P_k @ J_A.T + Q
    error_cov = np.matmul(np.matmul(JA, error_cov), JA.T) + Q

    # Measurement Function
    # 我们的观测就停在x 和 y 和 theta上，我们无法观测速度
    # transform function from predicted state to predicted observation
    hx = np.array([x[0], x[1], x[2], x[3]])

    # every nth step pretend there is a measurement
    # 观测方程对四个state进行求导，f_1 = x_1, f_2 = x_2
    # 那么对观测方程的'线性化'求导就会是
    # [d_f_1/d_x_1, d_f_1/d_x_2, d_f_1/d_x_3, d_f_1/d_x_4]
    # [d_f_2/d_x_1, d_f_2/d_x_2, d_f_2/d_x_3, d_f_2/d_x_4]
    # 用这个来更新观测的噪声矩阵

    JH = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # 根据第一部分计算卡尔曼增益
    R = np.diag([.000001, .000001, .000001, .000001])
    S = np.matmul(np.matmul(JH, error_cov), JH.T) + R
    K = np.matmul(np.matmul(error_cov, JH.T), np.linalg.inv(S.astype('float64')))

    Z = np.array(m)
    # 计算残差
    y = Z - hx
    # 更新
    x = x + np.matmul(K, y)
    # Update the error covariance
    error_cov = np.matmul((I - np.matmul(K, JH)), error_cov)
    # 返回滤波后的u_k+1|k+1, cov_k+1|k+1
    return x, error_cov


def ekf_missed(m, x, filterstep, error_cov):
    I = np.eye(4)

    # Project the state ahead
    # g(x)的解析式
    x[0] = x[0] + filterstep * x[3] * np.cos(x[2])
    x[1] = x[1] + filterstep * x[3] * np.sin(x[2])
    # 保持角度在[-np.pi, np.pi]之间
    x[2] = (x[2] + np.pi) % (2.0 * np.pi) - np.pi  # 角度
    x[3] = x[3]  # 速度

    # Calculate the Jacobian of the Dynamic Matrix A
    # 线性化卡尔曼矩阵
    # g对x求导得到的
    a13 = filterstep * x[3] * np.sin(x[2])
    a14 = filterstep * np.cos(x[2])
    a23 = filterstep * x[3] * np.cos(x[2])
    a24 = filterstep * np.sin(x[2])
    JA = np.array([[1.0, 0.0, a13, a14],
                   [0.0, 1.0, a23, a24],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # Process noise cov
    Q = np.diag([0.05, 0.05, 0.05, 0.05]) * 0.001

    # Project the error covariance ahead
    # 更新噪声 J_A @ P_k @ J_A.T + Q
    error_cov = np.matmul(np.matmul(JA, error_cov), JA.T) + Q

    # Measurement Function
    # 我们的观测就停在x 和 y 和 theta上，我们无法观测速度
    # transform function from predicted state to predicted observation
    hx = np.array([x[0], x[1], x[3]])

    # every nth step pretend there is a measurement
    # 观测方程对四个state进行求导，f_1 = x_1, f_2 = x_2
    # 那么对观测方程的'线性化'求导就会是
    # [d_f_1/d_x_1, d_f_1/d_x_2, d_f_1/d_x_3, d_f_1/d_x_4]
    # [d_f_2/d_x_1, d_f_2/d_x_2, d_f_2/d_x_3, d_f_2/d_x_4]
    # 用这个来更新观测的噪声矩阵

    JH = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    # 根据第一部分计算卡尔曼增益
    R = np.diag([.000001, .000001, .000001])
    S = np.matmul(np.matmul(JH, error_cov), JH.T) + R
    K = np.matmul(np.matmul(error_cov, JH.T), np.linalg.inv(S.astype('float64')))

    Z = np.array(m)
    # 计算残差
    y = Z - hx
    # 更新
    x = x + np.matmul(K, y)
    # Update the error covariance
    error_cov = np.matmul((I - np.matmul(K, JH)), error_cov)
    # 返回滤波后的u_k+1|k+1, cov_k+1|k+1
    return x, error_cov