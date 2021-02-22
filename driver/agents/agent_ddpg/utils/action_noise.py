import numpy as np

"""
Python implementation of Ornstein-Uhlenbeck process for random noise generation
"""
class OUActionNoise(object):
    def __init__(self, mu, sigma = 0.2, theta=0.15, dt=1e-2, x_0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = np.ones(1) * sigma
        self.dt = dt
        self.x_0 = x_0
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_0 is not None:
            self.x_prev = self.x_0
        else:
            self.x_prev = np.zeros_like(self.mu)
