import numpy as np

class Environment:
    dt = 18
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
