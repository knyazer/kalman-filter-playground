import numpy as np

sigma = 0.03


class KalmanFilterBigState:
    # the observation covariance matrix
    R = np.eye(2, dtype=np.float64) * sigma * sigma

    # the state covariance matrix
    Q = np.eye(6, dtype=np.float64) * 1e-12  # In our case it is super non-gaussian, but we don't care lolz

    # the time difference between two steps, in ms
    dt = 18

    # the state transition matrix, 6x6
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0, dt * dt / 2.0, 0.0],
            [0.0, 1.0, 0.0, dt, 0.0, dt * dt / 2.0],
            [0.0, 0.0, 1.0, 0.0, dt, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # the observation model
    H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    # the state estimate
    state = np.zeros((6, 1), dtype=np.float64)

    # the state covariance matrix, assign some large value
    P = np.eye(6, dtype=np.float64)

    def __init__(self):
        pass

    def update_dt(self, dt):
        self.dt = dt
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0, dt * dt / 2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, dt * dt / 2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def reset(self):
        self.state = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)

    def update_hyps(self, cov1, cov2, cov3):
        # we assign cov1 to the first two states (we know them precisely, kinda), cov2 to the next two (speeds are less accurate) and cov3 to the last one (acceleration is kinda completely trashed)
        self.Q = np.array(
            [
                [cov1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, cov1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, cov2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, cov2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, cov3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, cov3],
            ],
            dtype=np.float64,
        )

    def predicted_state_estimate(self):
        return self.F @ self.state  # just one dynamics step

    def predicted_state_covariance(self):
        rescaled = self.F @ (self.P @ self.F.T)
        return rescaled + self.Q

    def predict(self):
        self.state = self.predicted_state_estimate()
        self.P = self.predicted_state_covariance()

    def innovation(self, z):
        out = z - (self.H @ self.state)
        return out

    def innovation_covariance(self):
        rescaled = self.H @ (self.P @ self.H.T)
        return rescaled + self.R

    def kalman_gain(self):
        return self.P @ self.H.T @ np.linalg.pinv(self.innovation_covariance())

    def update(self, z):
        z = z.reshape((2, 1))
        self.predict()
        self.state = self.state + (self.kalman_gain() @ self.innovation(z))
        self.P = (np.eye(6, dtype=np.float64) - (self.kalman_gain() @ self.H)) @ self.P


class KalmanFilterSmallState:
    # the observation covariance matrix
    R = np.eye(2, dtype=np.float64) * sigma * sigma

    # the state covariance matrix
    Q = np.eye(4, dtype=np.float64) * 1e-7  # In our case it is super non-gaussian, but we don't care lolz

    # the time difference between two steps, in ms
    dt = 18

    # the state transition matrix, 4x4
    F = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # the observation model
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    # the state estimate
    state = np.zeros((4, 1), dtype=np.float64)

    # the state covariance matrix, assign some large value
    P = np.eye(4, dtype=np.float64)

    def __init__(self):
        pass

    def update_dt(self, dt):
        self.dt = dt
        self.F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def update_hyps(self, cov1, cov2):
        # we assign cov1 to the first two states, and cov2 to the last two states
        self.Q = np.array(
            [[cov1, 0.0, 0.0, 0.0], [0.0, cov1, 0.0, 0.0], [0.0, 0.0, cov2, 0.0], [0.0, 0.0, 0.0, cov2]],
            dtype=np.float64,
        )
        # cov2 should be bigger than cov1 probably

    def reset(self):
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def predicted_state_estimate(self):
        return self.F @ self.state  # just one dynamics step

    def predicted_state_covariance(self):
        rescaled = self.F @ (self.P @ self.F.T)
        return rescaled + self.Q

    def predict(self):
        self.state = self.predicted_state_estimate()
        self.P = self.predicted_state_covariance()

    def innovation(self, z):
        out = z - (self.H @ self.state)
        return out

    def innovation_covariance(self):
        rescaled = self.H @ (self.P @ self.H.T)
        return rescaled + self.R

    def kalman_gain(self):
        return self.P @ self.H.T @ np.linalg.pinv(self.innovation_covariance())

    def update(self, z):
        z = z.reshape((2, 1))
        self.predict()
        self.state = self.state + (self.kalman_gain() @ self.innovation(z))
        self.P = (np.eye(4, dtype=np.float64) - (self.kalman_gain() @ self.H)) @ self.P
