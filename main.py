import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import trange


class GreyScottSimulator:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.grid_A = np.ones((X, Y))
        self.grid_B = np.zeros((X, Y))
        self.lp_kernel = np.array(
            [[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]]
        )
        self.rng = np.random.default_rng()
        return

    def seed(self, n_seeds, diameter):
        for _ in range(n_seeds):
            x = int(self.rng.random() * self.X) - diameter
            y = int(self.rng.random() * self.Y) - diameter
            for i in range(diameter):
                for j in range(diameter):
                    self.grid_B[x + i][y + j] = 1
        return

    def laplace(self, grid):
        return ndimage.convolve(grid, self.lp_kernel, mode="reflect", cval=0)

    def update(self, dA, dB, fA, kB):
        a = self.grid_A
        b = self.grid_B
        self.grid_A = a + (dA * self.laplace(a) - a * b * b + fA * (1 - a))
        self.grid_B = b + (dB * self.laplace(b) + a * b * b - (kB + fA) * b)
        return

    def simulate(self, iterations, dA, dB, fA, kB):
        for _ in trange(iterations):
            self.update(dA, dB, fA, kB)
        return

    def show(self):
        plt.subplot(111)
        plt.imshow(self.grid_B, cmap="Greys", interpolation="none")
        plt.axis("off")
        plt.show()
        return


if __name__ == "__main__":
    g = GreyScottSimulator(400, 400)
    g.seed(n_seeds=50, diameter=50)
    g.simulate(3000, dA=1, dB=0.5, fA=0.055, kB=0.062)
    g.show()
