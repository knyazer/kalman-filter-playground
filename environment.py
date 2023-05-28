import numpy as np
from extractors import *
import os


class EnvironmentBase:
    dt = 18.0
    speed = 1.0
    noise_sigma = 0.02
    size = 12.2

    def wants_reset(self):
        return False

    def reset(self):
        pass

    def set_speed(self, speed):
        self.speed = speed

    def get_true_pos(self):
        return self.pos

    def get_true_vel(self):
        return self.vel

    def __init__(self):
        self.pos = np.array([0, 0], dtype=np.float32)

    def set_noise_sigma(self, sigma):
        self.noise_sigma = sigma

    def step(self):
        self.resolve_collisions()

        self.vel = self.vel / (np.linalg.norm(self.vel) + 1e-6) * self.speed
        self.pos = self.pos + self.vel * self.dt / 1000

    def observe(self):
        # return ball position, +- gaussian noise
        return self.pos + np.random.normal(0, self.noise_sigma, 2)

    def resolve_collisions(self):
        # check if ball goes out of the [-1, 1] range by x or y
        boundary = self.size / 2
        # and move it back
        if self.pos[0] < -boundary:
            self.pos[0] = -boundary
            self.vel[0] *= -1
        elif self.pos[0] > boundary:
            self.pos[0] = boundary
            self.vel[0] *= -1

        if self.pos[1] < -boundary:
            self.pos[1] = -boundary
            self.vel[1] *= -1
        elif self.pos[1] > boundary:
            self.pos[1] = boundary
            self.vel[1] *= -1


class GravityEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()            
        self.pos = np.array([0.0, -0.6], dtype=np.float32)
        self.vel = np.array([0.2, 0.0], dtype=np.float32)

    def step(self):
        super().step()

        self.vel[1] = self.vel[1] + 0.001 * self.dt * self.speed


class StraightEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()
        self.pos = np.array([0, 0], dtype=np.float32)
        self.vel = np.array([np.pi / 4, -np.e / 7], dtype=np.float32)


class CirclesSmallEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()
        self.pos = np.array([0.3, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 1.0], dtype=np.float32)

    def step(self):
        super().step()

        # normalize the position to 0.3
        self.pos = self.pos / np.linalg.norm(self.pos) * 0.3
        # velocity perp to the position
        self.vel = np.array([-self.pos[1], self.pos[0]], dtype=np.float32)
        self.vel = self.vel / (np.linalg.norm(self.vel) + 1e-6) * self.speed


class CirclesBigEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()
        self.pos = np.array([0.8, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 5.0], dtype=np.float32)

    def step(self):
        super().step()

        # normalize the position to 0.8
        self.pos = self.pos / np.linalg.norm(self.pos) * 0.8
        # velocity perp to the position
        self.vel = np.array([-self.pos[1], self.pos[0]], dtype=np.float32)
        self.vel = self.vel / np.linalg.norm(self.vel) * self.speed


env_names = ['gravity', 'straight', 'circles_small', 'circles_big', 'extracted']

class ExtractedEnvironment(EnvironmentBase):
    def __init__(self):
        super().__init__()

        self.all_data = []
        for name in os.listdir('data'):
            self.all_data.append(extract('data/' + name))
        
        self.current_piece = 0
        self.data = self.all_data[self.current_piece]

        self.data[:, 1:] = self.data[:, 1:] / 1000.0 # in meters
        self.pos = self.data[0][1:]
        self.vel = np.zeros(2, dtype=np.float32)
        self.current_timestamp_index = 0
        self.want_reset = False

    def step(self):
        super().step()

        self.current_timestamp_index += 1
        if self.current_timestamp_index >= len(self.data):
            self.current_timestamp_index = len(self.data) - 1
            self.want_reset = True
        self.pos = self.data[self.current_timestamp_index][1:]
        self.dt = (self.data[self.current_timestamp_index][0] - self.data[self.current_timestamp_index - 1][0]) * 1000

    def wants_reset(self):
        return self.want_reset

    def reset(self):
        self.current_piece = (self.current_piece + 1) % len(self.all_data)
        self.data = self.all_data[self.current_piece]
        self.data[:, 1:] = self.data[:, 1:] / 1000.0 # in meters

        self.current_timestamp_index = 0
        self.want_reset = False
        self.pos = self.data[0][1:]

    def observe(self):
        return self.pos


class Environment:
    def __init__(self, name):
        self.set_env_name(name)

    def set_env_name(self, name):
        if name == 'straight':
            self.env = StraightEnvironment()
        elif name == 'circles_small':
            self.env = CirclesSmallEnvironment()
        elif name == 'circles_big':
            self.env = CirclesBigEnvironment()
        elif name == 'gravity':
            self.env = GravityEnvironment()
        elif name == 'extracted':
            self.env = ExtractedEnvironment()
        else:
            raise ValueError('Unknown environment name: ' + name)

    def set_env_index(self, index):
        global env_names
        name = env_names[index]
        self.set_env_name(name)

    def step(self):
        self.env.step()

    def observe(self):
        return self.env.observe()

    def set_speed(self, speed):
        self.env.set_speed(speed)

    def set_noise_sigma(self, sigma):
        self.env.set_noise_sigma(sigma)

    def get_true_pos(self):
        return self.env.get_true_pos()

    def get_true_vel(self):
        return self.env.get_true_vel()

    def get_size(self):
        return self.env.size

    def get_variants():
        return env_names

    def get_dt(self):
        return self.env.dt / 1000.0

    def wants_reset(self):
        return self.env.wants_reset()

    def reset(self):
        self.env.reset()
