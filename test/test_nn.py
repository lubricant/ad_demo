"""
============================
Circles, Wedges and Polygons
============================
"""

import numpy as np

from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt
import matplotlib.colors as clr

fig, ax = plt.subplots()

N = 60
ax_min, ax_max = -3, 3
ax_step = (ax_max-ax_min) / N

tick = np.arange(ax_min, ax_max, ax_step)
patches = PatchCollection([Rectangle((x, y), ax_step, ax_step) for y in tick for x in tick])

buf = []
scatter = (
    ax.scatter([], [], color='indianred', marker='o', linewidths=5, animated=True),
    ax.scatter([], [], color='seagreen', marker='o', linewidths=5, animated=True))


def init_data():
    return [
        (np.array([-0.4326, 1.1909]), 1),
        (np.array([3.0, 4.0]), 1),
        (np.array([0.1253, -0.0376]), 1),
        (np.array([0.2877, 0.3273]), 1),
        (np.array([-1.1465, 0.1746]), 1),
        (np.array([1.8133, 1.0139]), 0),
        (np.array([2.7258, 1.0668]), 0),
        (np.array([1.4117, 0.5593]), 0),
        (np.array([4.1832, 0.3044]), 0),
        (np.array([1.8636, 0.1677]), 0),
        (np.array([0.5, 3.2]), 1),
        (np.array([0.8, 3.2]), 1),
        (np.array([1.0, -2.2]), 1)]


def circle_data():
    data = []
    for i in range(30):
        t = (np.random.rand() - 0.5) * np.pi
        r = (np.random.rand() - 0.5) * 1.5
        data.append((r * np.array([np.sin(t), np.cos(t)]), 1))

    for i in range(30):
        t = (np.random.rand() - 0.5) * np.pi
        r = (np.random.rand() - 0.5)
        r += np.sign(r) * 2.5
        data.append((r * np.array([np.sin(t), np.cos(t)]), 0))
    return data


def spiral_data():
    data = []
    for i in range(30):
        r = i / 30. + (np.random.rand() - 0.5) / 0.5
        t = 1.25 * i / 30.  * np.pi + (np.random.rand() - 0.5) / 0.5
        data.append((r * np.array([np.sin(t), np.cos(t)]), 1))

    # for i in range(30):
    #     r = i / 30. + (np.random.rand() - 0.5) / 0.5
    #     t = 1.25 * i / 30. * np.pi + np.pi + (np.random.rand() - 0.5) / 0.5
    #     data.append((r * np.array([np.sin(t), np.cos(t)]), 0))

    return data


def init_plot():

    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)

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

    for sca in scatter:
        sca.set_offsets([[], []])

    buf[:] = spiral_data()

    return (patches,) + scatter


def update_plot(i):

    colors = [clr.cnames['pink' if x == y else 'lightgreen'] for y in tick for x in tick]
    patches.set_color(colors)

    if buf:

        lab_0_x, lab_0_y = [], []
        lab_1_x, lab_1_y = [], []
        for feature, label in buf:
            if label:
                lab_1_x.append(feature[0])
                lab_1_y.append(feature[1])
            else:
                lab_0_x.append(feature[0])
                lab_0_y.append(feature[1])

        scatter[0].set_offsets([lab_0_x, lab_0_y])
        scatter[1].set_offsets([lab_1_x, lab_1_y])

        buf[:] = []

    return (patches,) + scatter


if __name__ == '__main__':

    ani = animation.FuncAnimation(fig, update_plot, init_func=init_plot, interval=5, blit=True)
    plt.show()
