import cv2 as cv
import numpy as np
from params import dt, SCREEN_SIZE, HALF_SCREEN_SIZE

def resize(x, env_size = 12.5):
    return int(x / (env_size / 2) * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE)

def draw_trace(img, trace, color, width, offset, roi_size):
    for i in range(len(trace) - 1):
        cv.line(
            img,
            (
                resize(trace[i][0] - offset[0], roi_size),
                resize(trace[i][1] - offset[1], roi_size),
            ),
            (
                resize(trace[i + 1][0] - offset[0], roi_size),
                resize(trace[i + 1][1] - offset[1], roi_size),
            ),
            color,
            width,
        )
    return img


def dynamics(state):
    if state.shape[0] == 4:
        x, y, vx, vy = state
        x += vx * dt
        y += vy * dt
        return np.array([x, y, vx, vy])
    elif state.shape[0] == 6:
        x, y, vx, vy, ax, ay = state
        x += vx * dt + 0.5 * ax * dt * dt
        y += vy * dt + 0.5 * ay * dt * dt
        vx += ax * dt
        vy += ay * dt
        return np.array([x, y, vx, vy, ax, ay])

def make_pred(state, dt):
    state = state.copy()
    if state.shape[0] == 4:
        x, y, vx, vy = state
        x += vx * dt
        y += vy * dt
        return np.array([x, y, vx, vy])
    elif state.shape[0] == 6:
        x, y, vx, vy, ax, ay = state
        x += vx * dt + 0.5 * ax * dt * dt
        y += vy * dt + 0.5 * ay * dt * dt
        vx += ax * dt
        vy += ay * dt
        return np.array([x, y, vx, vy, ax, ay])

    return x

def make_trace(state, length):
    x = state.copy()
    trace = []
    for i in range(length):
        trace.append(x[:2])
        x = dynamics(x)
    return trace

camera_viewpoint = np.zeros(2, dtype=np.float32)
def render(traces, timestamp, props, roi_size=3, abs_render=False, dt=0.018, length=60):
    img = np.zeros((SCREEN_SIZE, SCREEN_SIZE, 3), dtype=np.uint8)
    img[:, :, 1] = 80
    now = traces["Timestamp"].index(timestamp) + 1
    ball_pos = traces["True"][now - 1]

    global camera_viewpoint
    camera_viewpoint = ball_pos[:2] * 0.1 + 0.9 * camera_viewpoint
    if abs_render:
        camera_viewpoint = np.zeros(2, dtype=np.float32)
        roi_size = 12.5

    low = np.round((camera_viewpoint - roi_size / 2 - 0.5) * 4)
    high = np.round((camera_viewpoint + roi_size / 2 + 0.5) * 4)
    sz = int(high[0] - low[0])
    low = low / 4
    high = high / 4
    for x in np.linspace(low[0], high[0], sz):
        for y in np.linspace(low[1], high[1], sz):
            # draw a small dark-green cross
            cv.line(
                img,
                (
                    resize(x - camera_viewpoint[0] - 0.01, roi_size),
                    resize(y - camera_viewpoint[1], roi_size),
                ),
                (
                    resize(x - camera_viewpoint[0] + 0.01, roi_size),
                    resize(y - camera_viewpoint[1], roi_size),
                ),
                (0, 50, 0),
                3
            )

    # draw the ball
    cv.circle(
        img,
        (
            resize(ball_pos[0] - camera_viewpoint[0], roi_size),
            resize(ball_pos[1] - camera_viewpoint[1], roi_size),
        ),
        10,
        (0, 0, 255),
        -1,
    )


    # draw the traces
    low = max(now - length, 0)
    high = min(now + length, len(traces["True"]))
    if props[0] == 1:
        img = draw_trace(img, traces["Noise"][low:now], (0, 0, 0), 1, camera_viewpoint, roi_size)
    if props[1] == 1:
        img = draw_trace(img, traces["K6"][low:now], (0, 190, 250), 2, camera_viewpoint, roi_size)
    if props[2] == 1:
        img = draw_trace(img, traces["K4"][low:now], (255, 100, 100), 2, camera_viewpoint, roi_size)
    if props[3] == 1:
        img = draw_trace(img, traces["True"][low:high], (255, 255, 255), 1, camera_viewpoint, roi_size)
        pos = traces["True"][now]
        cv.circle(img, (
            resize(pos[0] - camera_viewpoint[0], roi_size),
            resize(pos[1] - camera_viewpoint[1], roi_size),
        ), 3, (255, 255, 255), -1)
    if props[4] == 1:
        tr = make_trace(traces["K6-full"][now - 1], length)
        img = draw_trace(img, tr, (50, 250, 255), 1, camera_viewpoint, roi_size)
        pos = make_pred(traces["K6-full"][now - 1], dt)
        cv.circle(img, (
            resize(pos[0] - camera_viewpoint[0], roi_size),
            resize(pos[1] - camera_viewpoint[1], roi_size),
        ), 3, (0, 200, 200), -1)
    if props[5] == 1:
        tr = make_trace(traces["K4-full"][now - 1], length)
        img = draw_trace(img, tr, (255, 150, 150), 1, camera_viewpoint, roi_size)
        pos = make_pred(traces["K4-full"][now - 1], dt)
        cv.circle(img, (
            resize(pos[0] - camera_viewpoint[0], roi_size),
            resize(pos[1] - camera_viewpoint[1], roi_size),
        ), 3, (200, 100, 110), -1)

    return img
