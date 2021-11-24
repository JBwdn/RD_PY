"""
Reaction diffusion visualisation
    Jake Bowden - 2021

Usage: "python main.py -m p" to preview the animation.
       "python main.py -m s" to save the animation to a gif

Dependencies:
    - numpy
    - scipy
    - matplotlib
"""


from typing import Tuple
import math
import argparse

import numpy as np
from scipy import ndimage

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Parameters:
SIZE = 400
FPS = 15
LENGTH = 20
FILE_NAME = "wallpaper.gif"

# Kernel for applying the Laplace transform:
LP_KERNEL = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])


def laplace(grid: np.ndarray) -> ndimage:
    """
    Using scipy ndimage class, calculate the Laplace convolution using LP_KERNEL
    """
    return ndimage.convolve(grid, LP_KERNEL, mode="constant", cval=0)


class GreyScottSimulator:
    """
    Simulate 2 component reaction diffusion system with animation
    See:
        - Turing bifurcation patterns (The Chemical Basis of Morphogenesis, 1952)
        - Grey & Scott method (doi.org/10.1016/0009-2509(83)80132-8, 1983)
    """

    def __init__(self, X: int, Y: int) -> None:
        self.X = X
        self.Y = Y

        # Default simulation parameters:
        self.dA = 1.0
        self.dB = 0.6
        self.fA = 0.031
        self.kB = 0.058
        self.mod = False

        self.grid_A = np.ones((X, Y))
        self.grid_B = np.zeros((X, Y))

        self.i = 0
        self._loop_n = 1
        self._rng = np.random.default_rng()

    def set_mod(self, fA_func: callable, kB_func: callable):
        """
        Activate modulation and set functions to control fA and kB
        Function signature: f(param,frame) = new_param
        eg: dfA = lambda i : sin(i) / 1000
        """
        self.mod = True
        self.fA_mod_func = fA_func
        self.kB_mod_func = kB_func

    def seed(self, n_seeds: int, diameter: int) -> None:
        """
        Seed the simulation space with regions of high [B]
        If only seeding once, place in the centre.
        TODO: make circular over squares?
        """
        if n_seeds == 1:
            x = self.X // 2
            y = self.Y // 2
            for i in range(diameter):
                for j in range(diameter):
                    self.grid_B[(x - diameter // 2) + i][(y - diameter // 2) + j] = 1
        else:
            for _ in range(n_seeds):
                x = int(self._rng.random() * self.X) - diameter
                y = int(self._rng.random() * self.Y) - diameter
                for i in range(diameter):
                    for j in range(diameter):
                        self.grid_B[x + i][y + j] = 1

    def step(self) -> None:
        """
        Step the simulation and calculate [A] and [B]
        TODO: port to torch tensors? should be faster and can run on CUDA
        """
        a = self.grid_A
        b = self.grid_B
        self.grid_A = a + (self.dA * laplace(a) - a * b * b + self.fA * (1 - a))
        self.grid_B = b + (self.dB * laplace(b) + a * b * b - (self.kB + self.fA) * b)

    def simulate(self, iterations: int) -> None:
        """
        Run the simulation for a number of iterations
        """
        for _ in range(iterations):
            self.step()
            self.i += 1

    def show(self) -> None:
        """
        Call pyplot.show() function to draw the current state of the simulation
        """
        plt.subplot(111)
        plt.imshow(self.grid_B, cmap="turbo", interpolation="none")
        plt.axis("off")
        plt.show()

    def modulate(self):
        """
        Modify fA and kB parameters programatically
        """
        self.fA += self.fA_mod_func(self.i)
        self.kB += self.kB_mod_func(self.i)

    def animate(self, framerate: int, length: int, step_size: int) -> FuncAnimation:
        """
        Generate a matplotlib FuncAnimation
        """
        anim_fig, anim_ax = plt.subplots()
        anim_im = anim_ax.imshow(self.grid_B, cmap="Purples")
        blank_board = np.zeros_like(self.grid_B)
        plt.axis("off")

        def animation_init() -> Tuple[matplotlib.image.AxesImage, None]:
            """
            Initialise the animation with a blank board
            """
            anim_im.set_data(blank_board)
            return (anim_im,)

        def animation_update(
            frame: int,
        ) -> Tuple[matplotlib.image.AxesImage, None]:
            """
            Generate each frame of the animation using the simulate method
            TODO: Move fA/kB modulation to be a function of iteration rather than frame...
                  - Make new method or add to self.update()
            """
            print(f"Rendering frame: {frame + 1}/{length}", end="\r", flush=True)

            # Hacky shift of grid B for shadow effect:
            B = np.round(self.grid_B + 0.3)
            B_shift = np.pad(B[4:, 4:], 2)
            anim_im.set_data((B * 2 + B_shift) / 2)

            if self.mod:
                self.modulate()

            self.simulate(step_size)

            # looping, seed with an additional point per loop:
            if self.grid_B.sum() < 1e-4:
                self._loop_n += 1
                self.seed(self._loop_n, 10)

            return (anim_im,)

        rd_animation = FuncAnimation(
            anim_fig,
            animation_update,
            init_func=animation_init,
            frames=length,
            interval=framerate,
            blit=True,
        )
        return rd_animation


if __name__ == "__main__":
    # Parse command line arguments:
    parser = argparse.ArgumentParser(description="Reaction Diffusion Sim:")
    parser.add_argument(
        "-m", type=str, help="Script mode: (p)review or (s)ave", required=True
    )
    args = parser.parse_args()

    # Set up simulator:
    gs = GreyScottSimulator(1080,1920)
    gs.seed(n_seeds=30, diameter=10)

    # Modulation functions:
    dfA = lambda i: math.sin((i * 500) / 13) / 1000
    dkB = lambda i: math.cos((i * 500) / 15) / 1000
    #gs.set_mod(dfA, dkB)

    # Set up animation:
    anim = gs.animate(framerate=FPS, length=LENGTH, step_size=30)

    # Preview or save mode:
    if args.m == "p":
        print("Previewing animation...")
        plt.show()
        print("Animation closed")
    elif args.m == "s":
        print("Saving animation...")
        anim.save(FILE_NAME, fps=20)
        print(f"Animation saved at {FILE_NAME}")
