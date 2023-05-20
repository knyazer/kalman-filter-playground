import cv2 as cv
import numpy as np
from params import dt, SCREEN_SIZE, HALF_SCREEN_SIZE



def draw_trace(img, trace, color, width):
    for i in range(len(trace) - 1):
        cv.line(
            img,
            (
                int(trace[i][0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
                int(trace[i][1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
            ),
            (
                int(trace[i + 1][0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
                int(trace[i + 1][1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
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

def make_trace(state, length):
    x = state.copy()
    trace = []
    for i in range(length):
        trace.append(x[:2])
        x = dynamics(x)
    return trace

def render(traces, timestamp, props, length=60):
    img = np.zeros((SCREEN_SIZE, SCREEN_SIZE, 3), dtype=np.uint8)
    img[:, :, 1] = 80
    now = traces["Timestamp"].index(timestamp) + 1
    ball_pos = traces["True"][now - 1]
    # draw the ball
    cv.circle(
        img,
        (
            int(ball_pos[0] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
            int(ball_pos[1] * HALF_SCREEN_SIZE + HALF_SCREEN_SIZE),
        ),
        10,
        (0, 0, 255),
        -1,
    )
    # draw the traces
    low = max(now - length, 0)
    if props[0] == 1:
        img = draw_trace(img, traces["Noise"][low:now], (0, 0, 0), 1)
    if props[1] == 1:
        img = draw_trace(img, traces["K6"][low:now], (0, 190, 250), 2)
    if props[2] == 1:
        img = draw_trace(img, traces["K4"][low:now], (255, 100, 100), 2)
    if props[3] == 1:
        img = draw_trace(img, traces["True"][low:now], (255, 255, 255), 1)
    if props[4] == 1:
        tr = make_trace(traces["K6-full"][now - 1], length)
        img = draw_trace(img, tr, (50, 250, 255), 1)
    if props[5] == 1:
        tr = make_trace(traces["K4-full"][now - 1], length)
        img = draw_trace(img, tr, (255, 150, 150), 1)

    return img
