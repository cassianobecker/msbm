import numpy as np
import matplotlib.pyplot as plt


def plot_elbos(elbos, par):

    nr = 2
    nc = np.ceil(len(elbos.keys())/2.)

    i = 0

    for key in elbos.keys():
        i = i + 1
        ax = plt.subplot(nr, nc, i)
        plt.plot(elbos[key])
        k = len(elbos[key])
        ax.set_title(key)
        plt.plot(range(k), par['elbos0'][key][0] * np.ones(k))
