"""
============================

============================
"""

import numpy as np

from matplotlib import animation

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, RadioButtons

import matplotlib.pyplot as plt
import matplotlib.colors as clr

N = 40
ax_min, ax_max = -4, 4
ax_step = (ax_max-ax_min) / N

tick = np.arange(ax_min, ax_max+ax_step, ax_step)
patches = PatchCollection([Rectangle((x, y), ax_step, ax_step) for y in tick for x in tick])

buf = []
data = []
scatter = []
buttons = []


def init_data(_):
    buf[:] = [
        (np.array([-0.4326, 1.1909]), 1),
        (np.array([3.0, 4.0]), 1),
        (np.array([0.1253, -0.0376]), 1),
        (np.array([0.2877, 0.3273]), 1),
        (np.array([-1.1465, 0.1746]), 1),
        (np.array([1.8133, 1.0139]), 0),
        (np.array([2.7258, 1.0668]), 0),
        (np.array([1.4117, 0.5593]), 0),
        (np.array([3.9832, 0.3044]), 0),
        (np.array([1.8636, 0.1677]), 0),
        (np.array([0.5, 3.2]), 1),
        (np.array([0.8, 3.2]), 1),
        (np.array([1.0, -2.2]), 1)]


def circle_data(_):
    data = []
    for i in range(50):
        t = (np.random.rand() - 0.5) * np.pi
        r = (np.random.rand() - 0.5) * 2.
        data.append((r * np.array([np.sin(t), np.cos(t)]), 1))

    for i in range(50):
        t = (np.random.rand() - 0.5) * np.pi
        r = (np.random.rand() - 0.5)
        r += np.sign(r) * 2.5
        data.append((r * np.array([np.sin(t), np.cos(t)]), 0))
    buf[:] = data


def spiral_data(_):
    data = []
    for theta in np.linspace(0, 3 * np.pi, num=50):
        r = (theta ** 2) / 25
        data.append((np.array([r * np.cos(theta), r * np.sin(theta)]), 0))

    for theta in np.linspace(0, 3*np.pi, num=50):
        r = (theta ** 2) / 25
        data.append((np.array([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)]), 1))
    buf[:] = data


def init_btn():
    buttons[:] = [
        Button(plt.axes([0.3, 0.03, 0.1, 0.03]), "reset"),
        Button(plt.axes([0.45, 0.03, 0.1, 0.03]), "circle"),
        Button(plt.axes([0.6, 0.03, 0.1, 0.03]), "spiral")]

    buttons[0].on_clicked(init_data)
    buttons[1].on_clicked(circle_data)
    buttons[2].on_clicked(spiral_data)


class ModelRender(object):

    fig, ax = plt.subplots()

    def __init__(self, model):
        self.model = model
        self.ani = animation.FuncAnimation(
            self.fig, lambda i: self.update_plot(), init_func=lambda: self.init_plot(), interval=10, blit=True)
        plt.show()

    def init_plot(self):

        plt.xlim(ax_min, ax_max)
        plt.ylim(ax_min, ax_max)

        ax = self.ax
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xticks([])
        ax.set_yticks([])

        patches.set_alpha(0.5)
        patches.set_linewidth(0.01)
        ax.add_collection(patches)

        init_btn()

        return (patches,) + tuple(scatter)

    def update_plot(self):

        if buf:
            scat_pt = [[[], []], [[], []]]

            for feature, label in buf:
                scat_pt[label][0].append(feature[0])
                scat_pt[label][1].append(feature[1])

            ax = self.ax
            scatter[:] = [
              ax.scatter(scat_pt[0][0], scat_pt[0][1], color='indianred', marker='o', linewidths=5),
              ax.scatter(scat_pt[1][0], scat_pt[1][1], color='seagreen', marker='o', linewidths=5)]

            data[:] = buf
            buf[:] = []

        model = self.model
        loss = model.update(data)

        if loss is not None:
            for i in range(10):
                loss += model.update(data)

        if loss is not None:
            print(loss/10)

        colors = [clr.cnames['lightcoral' if model.predict(x, y) else 'lightgreen'] for y in tick for x in tick]
        patches.set_color(colors)

        # print([model.predict(x, y) for y in tick for x in tick])

        return (patches,) + tuple(scatter)


if __name__ == '__main__':
    from .nn_model import BinaryClassifierModel
    ModelRender(BinaryClassifierModel())
