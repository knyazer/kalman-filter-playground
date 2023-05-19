import cv2 as cv
import numpy as np

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
