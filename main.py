import cv2 as cv
import sys
import numpy as np

global_dt = 18

class KalmanFilterBigState:
    # the observation covariance matrix
    R = np.eye(2) * 0.04

    # the state covariance matrix
    Q = np.eye(6) * 1e-12 # In our case it is super non-gaussian, but we don't care lolz

    # the time difference between two steps, in ms
    dt = 18

    # the state transition matrix, 6x6
    F = np.array([[1.0, 0.0, dt, 0.0, dt * dt / 2, 0.0],
                  [0.0, 1.0, 0.0, dt, 0.0, dt * dt / 2],
                  [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    # the observation model
    H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # the state estimate
    state = np.zeros((6, 1), dtype=np.float32)

    # the state covariance matrix, assign some large value
    P = np.eye(6, dtype=np.float32)

    def __init__(self):
        pass

    def update_dt(self, dt):
        self.dt = dt
        self.F = np.array([[1.0, 0.0, dt, 0.0, dt * dt / 2, 0.0],
                           [0.0, 1.0, 0.0, dt, 0.0, dt * dt / 2],
                           [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def reset(self):
        self.state = np.zeros((6, 1))
        self.P = np.eye(6)
    
    def update_hyps(self, cov1, cov2, cov3):
        # we assign cov1 to the first two states (we know them precisely, kinda), cov2 to the next two (speeds are less accurate) and cov3 to the last one (acceleration is kinda completely trashed)
        self.Q = np.array([[cov1, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, cov1, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, cov2, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, cov2, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, cov3, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, cov3]], dtype=np.float32)

    def predicted_state_estimate(self):
        return self.F @ self.state # just one dynamics step

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
        self.P = (np.eye(6) - (self.kalman_gain() @ self.H)) @ self.P

class KalmanFilterSmallState:
    # the observation covariance matrix
    R = np.eye(2) * 0.04

    # the state covariance matrix
    Q = np.eye(4) * 1e-7 # In our case it is super non-gaussian, but we don't care lolz

    # the time difference between two steps, in ms
    dt = 18

    # the state transition matrix, 4x4
    F = np.array([[1.0, 0.0, dt, 0.0],
                  [0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    # the observation model
    H = np.array([[1.0, 0.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    # the state estimate
    state = np.zeros((4, 1), dtype=np.float32)

    # the state covariance matrix, assign some large value
    P = np.eye(4, dtype=np.float32)

    def __init__(self):
        pass

    def update_dt(self, dt):
        self.dt = dt
        self.F = np.array([[1.0, 0.0, dt, 0.0],
                  [0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    def update_hyps(self, cov1, cov2):
        # we assign cov1 to the first two states, and cov2 to the last two states
        self.Q = np.array([[cov1, 0.0, 0.0, 0.0],
                           [0.0, cov1, 0.0, 0.0],
                           [0.0, 0.0, cov2, 0.0],
                           [0.0, 0.0, 0.0, cov2]], dtype=np.float32)
        # cov2 should be bigger than cov1 probably

    def reset(self):
        self.state = np.zeros((4, 1))
        self.P = np.eye(4)
    
    def predicted_state_estimate(self):
        return self.F @ self.state # just one dynamics step

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
        self.P = (np.eye(4) - (self.kalman_gain() @ self.H)) @ self.P

class Environment:
    dt = global_dt
    name = "straight"

    def __init__(self, name):
        if not name in ["circles-small", "circles-big", "straight"]:
            raise ValueError("Wrong environment name")
        self.ball_pos = np.array([0, 0], dtype=np.float32)
        self.ball_vel = np.array([0.9, 0.2], dtype=np.float32)
        self.name = name

        if name == "circles-big":
            self.ball_pos = np.array([0.8, 0.0], dtype=np.float32)
            self.ball_vel = np.array([0.0, 1.0], dtype=np.float32)
        if name == "circles-small":
            self.ball_pos = np.array([0.3, 0.0], dtype=np.float32)
            self.ball_vel = np.array([0.0, 1.0], dtype=np.float32)

    def step(self):
        if self.name == "straight":
            self.ball_pos = self.ball_pos + self.ball_vel * self.dt / 1000
            self.resolve_collisions()
        elif self.name == "circles-big" or self.name == "circles-small":
            self.ball_pos = self.ball_pos + self.ball_vel * self.dt / 1000
            self.resolve_collisions()
            # to go in circles we need to rotate the velocity vector
            # so that it is perpendicular to the position vector
            # and then we need to normalize it
            self.ball_vel = self.ball_pos / np.linalg.norm(self.ball_pos)
            self.ball_vel = np.array([-self.ball_vel[1], self.ball_vel[0]], dtype=np.float32)

        if self.name == "circles-big":
            # normalize the position to 0.8
            self.ball_pos = self.ball_pos / np.linalg.norm(self.ball_pos) * 0.8
        if self.name == "circles-small":
            # normalize the position to 0.3
            self.ball_pos = self.ball_pos / np.linalg.norm(self.ball_pos) * 0.3


    def observe(self):
        # return ball position, +- gaussian noise
        return self.ball_pos + np.random.normal(0, 0.02, 2)

    def resolve_collisions(self):
        # check if ball goes out of the [-1, 1] range by x or y
        boundary = 0.9
        # and move it back
        if self.ball_pos[0] < -boundary:
            self.ball_pos[0] = -boundary
            self.ball_vel[0] *= -1
        elif self.ball_pos[0] > boundary:
            self.ball_pos[0] = boundary
            self.ball_vel[0] *= -1

        if self.ball_pos[1] < -boundary:
            self.ball_pos[1] = -boundary
            self.ball_vel[1] *= -1
        elif self.ball_pos[1] > boundary:
            self.ball_pos[1] = boundary
            self.ball_vel[1] *= -1

# now, we not only want to do a cute lil simulation, but we also want to tune the hyperparameters of kalman (Q matrix) so that it minimizes L2 error with the true one. One might see this as an automatic calibration of sorts: before the match we run a cute little 

SCREEN_SIZE = 700

HALF_SCREEN_SIZE = SCREEN_SIZE // 2

def draw_trace(img, trace, color, width):
    for i in range(len(trace) - 1):
        cv.line(img, (int(trace[i][0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE), int(trace[i][1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE)), (int(trace[i + 1][0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE), int(trace[i + 1][1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE)), color, width)
    return img

def render(traces, timestamp):
    img = np.zeros((SCREEN_SIZE, SCREEN_SIZE, 3), dtype=np.uint8)
    img[:,:,1] = 80
    now = traces["Timestamp"].index(timestamp)
    ball_pos = traces["True"][now]
    # draw the ball
    cv.circle(img, (int(ball_pos[0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE), int(ball_pos[1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE)), 10, (0, 0, 255), -1)
    # draw the traces
    low = max(now - 60, 0)
    img = draw_trace(img, traces["Noise"][low:now], (0, 0, 0), 1)
    img = draw_trace(img, traces["K4"][low:now], (255, 0, 0), 2)
    img = draw_trace(img, traces["K6"][low:now], (0, 190, 250), 2)
    img = draw_trace(img, traces["True"][low:now], (255, 255, 255), 1)

    return img

env = Environment(["circles-small", "circles-big", "straight"][2])
def main():
    sst = env.observe()
    timestamp = 0
    traces = {"K4": [sst], "K6": [sst], "Noise": [sst], "True": [sst], "Timestamp": [timestamp]}
    filters = {"K4": KalmanFilterSmallState(), "K6": KalmanFilterBigState()}

    pause = False

    while True:
        # simulate the "next" step
        if not pause:
            timestamp += 1

        if not (timestamp in traces["Timestamp"]):
            env.step()

            # observe the ball position
            z = env.observe()

            for filt in filters.values():
                filt.update_dt(env.dt)
                filt.update(z)

            traces["K4"].append(filters["K4"].state)
            traces["K6"].append(filters["K6"].state)
            traces["Noise"].append(z)
            traces["True"].append(env.ball_pos)
            traces["Timestamp"].append(timestamp)

        # draw the trace, as a set of lines
        cv.imshow('world', render(traces, timestamp))

        # if space pressed, pause the simulation
        key = cv.waitKey(5)
        if key == ord(' '):
            pause = not pause
        # if escape pressed, exit
        if key == 27:
            break
        if key == ord('l') or key == ord('k'):
            pause = True
            timestamp += 1

        if key == ord('j') or key == ord('h'):
            pause = True
            timestamp -= 1


        # if window closed - exit
        if cv.getWindowProperty('world', cv.WND_PROP_VISIBLE) < 1:
            break

if __name__ == '__main__':
    # Check the first argument, and if it is a number, use it as an environment id
    if len(sys.argv) > 1:
        try:
            env = Environment(["circles-small", "circles-big", "straight"][int(sys.argv[1])])
        except:
            print("Invalid environment id")
            exit(1)
    main()
