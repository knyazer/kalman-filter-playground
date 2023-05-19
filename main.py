import cv2 as cv
import sys
import numpy as np
from filters import KalmanFilterSmallState, KalmanFilterBigState
from environment import Environment
from utils import render, draw_trace

env = Environment(["circles-small", "circles-big", "straight"][2])
def main():
    sst = env.observe()
    timestamp = 0
    traces = {"K4": [sst], "K6": [sst], "Noise": [sst], "True": [sst], "Timestamp": [timestamp]}
    filters = {"K4": KalmanFilterSmallState(), "K6": KalmanFilterBigState()}

    filters["K4"].update_hyps(1e-9, 1e-7)
    filters["K6"].update_hyps(1e-10, 1e-10, 1e-10)

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
