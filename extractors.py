import numpy as np

def extract(file_name):
    file = np.load(file_name)
    x, y, t = file['x'], file['y'], file['t']

    mask = np.concatenate((t[:-1] != t[1:], [True]))

    x_filtered = x[mask]
    y_filtered = y[mask]
    t_filtered = t[mask]

    data = np.stack((t_filtered, x_filtered, y_filtered), axis=1)
    return data
