import numpy as np


class Environment:
    dt = 18.0
    name = "straight"
    speed = 1.0
    noise_sigma = 0.02

    all_names = ["circles-small", "circles-big", "straight", "gravity"]

    def set_speed(self, speed):
        self.speed = speed

    def __init__(self, name):
        if name not in self.all_names:
            raise ValueError(
                "Wrong environment name, should be one of circles-small, circles-big, straight or gravity"
            )
        self.ball_pos = np.array([0, 0], dtype=np.float32)
        self.ball_vel = np.array([np.pi / 4, -np.e / 7], dtype=np.float32)
        self.name = name

        if name == "circles-big":
            self.ball_pos = np.array([0.8, 0.0], dtype=np.float32)
            self.ball_vel = np.array([0.0, 5.0], dtype=np.float32)
        if name == "circles-small":
            self.ball_pos = np.array([0.3, 0.0], dtype=np.float32)
            self.ball_vel = np.array([0.0, 1.0], dtype=np.float32)
        if name == "gravity":
            self.ball_pos = np.array([0.0, -0.6], dtype=np.float32)
            self.ball_vel = np.array([0.2, 0.0], dtype=np.float32)

    def set_env_index(self, index):
        if index == None or index < 0 or index >= len(self.all_names):
            raise ValueError("Wrong environment index")
        self.__init__(self.all_names[index])

    def set_noise_sigma(self, sigma):
        self.noise_sigma = sigma

    def step(self):
        # norm the velocity and scale by speed
        self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.speed
        self.ball_pos = self.ball_pos + self.ball_vel * self.dt / 1000

        if self.name == "straight":
            self.resolve_collisions()
        elif self.name == "circles-big" or self.name == "circles-small":
            self.resolve_collisions()
            # to go in circles we need to rotate the velocity vector
            # so that it is perpendicular to the position vector
            # and then we need to normalize it
            self.ball_vel = self.ball_pos / np.linalg.norm(self.ball_pos)
            self.ball_vel = np.array([-self.ball_vel[1], self.ball_vel[0]], dtype=np.float32)
        elif self.name == "gravity":
            self.ball_vel[1] = self.ball_vel[1] + 0.001 * self.dt * self.speed
            self.resolve_collisions()

        if self.name == "circles-big":
            # normalize the position to 0.8
            self.ball_pos = self.ball_pos / np.linalg.norm(self.ball_pos) * 0.8
        if self.name == "circles-small":
            # normalize the position to 0.3
            self.ball_pos = self.ball_pos / np.linalg.norm(self.ball_pos) * 0.3

    def observe(self):
        # return ball position, +- gaussian noise
        return self.ball_pos + np.random.normal(0, self.noise_sigma, 2)

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
