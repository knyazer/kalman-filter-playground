import sys

import cv2 as cv
import numpy as np
from environment import Environment
from filters import KalmanFilterBigState, KalmanFilterSmallState
from utils import render


# tuning hyps, means that we are running simulation for lets say 1000 steps (18 seconds) and then compute integral of L1 error, which is effectively an area :) seems like L1 is more meaningful in this setting
def estimate(filt, steps, *args):
    env = Environment("gravity")
    filt.reset()
    filt.update_hyps(*args)
    # warm up
    for i in range(10):
        env.step()
        filt.update(env.observe())

    # run the simulation
    err = 0
    for i in range(steps):
        env.step()
        filt.update(env.observe())
        err += np.abs(env.ball_pos - filt.state[:2]).sum()
        # and also don't forget velocity!
        err += np.abs(env.ball_vel - filt.state[2:4]).sum() * 0.3
    filt.reset()

    return err


def tune(filt, num_of_args):
    final_hyps = []
    # estimate every error for logs of 1.3
    for i in range(num_of_args):
        hyp = 1e-2
        best_est = 10e50
        best_hyp = hyp
        while hyp > 1e-20:
            inp = final_hyps + [hyp] * (num_of_args - i)
            est = estimate(filt, 500, *inp)
            hyp /= 1.4
            if est < best_est:
                best_est = est
                best_hyp = hyp
        final_hyps.append(best_hyp)
        print("Best hyp for arg {} is {}".format(i, best_hyp))
    print("Best estimate is {}".format(best_est))

    filt.update_hyps(*final_hyps)


env = Environment("straight")


def main():
    traces = {}
    timestamp = 0
    def switch_env(x):
        env.set_env_index(x)
        env.step()
        
        sst = env.observe()
        full_state = np.zeros(6, dtype=np.float32)
        full_state[:2] = sst[:2]
        timestamp = 0
        traces["K4"] = [sst]
        traces["K6"] = [sst]
        traces["Noise"] = [sst]
        traces["True"] = [sst]
        traces["Timestamp"] = [timestamp]
        traces["K4-full"] = [full_state[:4]]
        traces["K6-full"] = [full_state]
        filters["K4"].reset()
        filters["K6"].reset()

    filters = {"K4": KalmanFilterSmallState(), "K6": KalmanFilterBigState()}
    switch_env(0)

    filters["K4"].update_hyps(3e-6, 3e-3)
    filters["K6"].update_hyps(1e-6, 1e-3, 6e-2)

    # trackbars for the filters
    cv.namedWindow("world")
    cv.createTrackbar("K4 position covariance", "world", 0, 200, lambda x: filters["K4"].update_hyps(100 * 0.8 ** x, filters["K4"].hyps[1]))
    cv.createTrackbar("K4 velocity covariance", "world", 0, 200, lambda x: filters["K4"].update_hyps(filters["K4"].hyps[0], 100 * 0.8 ** x))

    cv.createTrackbar("K6 position covariance", "world", 0, 200, lambda x: filters["K6"].update_hyps(100 * 0.8 ** x, filters["K6"].hyps[1], filters["K6"].hyps[2]))
    cv.createTrackbar("K6 velocity covariance", "world", 0, 200, lambda x: filters["K6"].update_hyps(filters["K6"].hyps[0], 100 * 0.8 ** x, filters["K6"].hyps[2]))
    cv.createTrackbar("K6 acceleration covariance", "world", 0, 200, lambda x: filters["K6"].update_hyps(filters["K6"].hyps[0], filters["K6"].hyps[1], 10 * 0.9 ** x))

    cv.createTrackbar("Speed multiplier", "world", 1, 100, lambda x: env.set_speed((x + 1) / 10.0))
    cv.createTrackbar("Choosing environment", "world", 0, 3, switch_env)
    cv.createTrackbar("Noise sigma, 0.9^x", "world", 1, 100, lambda x: env.set_noise_sigma(0.9 ** x))

    # load the old trackbars positions stored in the file
    try:
        with open("trackbars.txt", "r") as f:
            for line in f:
                name, pos = line.split(";")
                cv.setTrackbarPos(name, "world", int(pos))
    except FileNotFoundError:
        pass


    props = [1, 1, 1, 1, 1, 1]

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
                filt.update(z)

            traces["K4"].append(filters["K4"].state[:2])
            traces["K6"].append(filters["K6"].state[:2])
            traces["Noise"].append(z)
            traces["True"].append(env.ball_pos)
            traces["Timestamp"].append(timestamp)
            traces["K4-full"].append(filters["K4"].state)
            traces["K6-full"].append(filters["K6"].state)

        # draw the trace, as a set of lines
        cv.imshow("world", render(traces, timestamp, props))

        # if space pressed, pause the simulation
        key = cv.waitKey(5)
        if key == ord(" "):
            pause = not pause
        # if escape pressed, exit
        if key == 27:
            break
        if key == ord("l") or key == ord("k"):
            pause = True
            timestamp += 1

        if key == ord("j") or key == ord("h"):
            pause = True
            timestamp -= 1

        if key == ord("1"):
            props[0] = 1 - props[0]
        if key == ord("2"):
            props[1] = 1 - props[1]
        if key == ord("3"):
            props[2] = 1 - props[2]
        if key == ord("4"):
            props[3] = 1 - props[3]
        if key == ord("5"):
            props[4] = 1 - props[4]
        if key == ord("6"):
            props[5] = 1 - props[5]

        # if window closed - exit
        if cv.getWindowProperty("world", cv.WND_PROP_VISIBLE) < 1:
            break

    # store all the trackbars positions in the file
    with open("trackbars.txt", "w") as f:
        for name in ["K4 position covariance", "K4 velocity covariance", "K6 position covariance", "K6 velocity covariance", "K6 acceleration covariance", "Speed multiplier", "Choosing environment", "Noise sigma, 0.9^x"]:
            f.write("{};{}\n".format(name, cv.getTrackbarPos(name, "world")))


if __name__ == "__main__":
    # Check the first argument, and if it is a number, use it as an environment id
    if len(sys.argv) > 1:
        try:
            env = Environment(["circles-small", "circles-big", "straight", "gravity"][int(sys.argv[1])])
        except ValueError:
            print("Invalid environment id")
            exit(1)
    main()
