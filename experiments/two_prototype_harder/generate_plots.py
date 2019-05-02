# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def main():
    stats_url = os.path.join('stats', 'stats_hard_inits.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #box plot y rand index
    #We create a list with the 6 boxes
    data1 = np.array(statistics['yrand_v2_nu_sp_']).flatten()
    data2 = np.array(statistics['yrand_v2_u_sp_']).flatten()
    data3 = np.array(statistics['yrand_v1_u_sp_']).flatten()
    data4 = np.array(statistics['yrand_v1_u_nsp_']).flatten()
    data5 = np.array(statistics['yrand_v2_u_nsp_']).flatten()

    data = [data1, data2, data3, data4, data5]
    plt.boxplot(data)
    plt.xticks(range(1,6),['sched2-nu-sp', 'sched2-u-sp', 'sched1-u-sp', 'sched1-u-nsp', 'sched2-u-nsp'], fontsize=7)
    plt.xlabel('Relevant Parameters')
    plt.ylabel('Adj. Rand Index of Y')
    plt.title("Rand Index for different Parameters.\n(50 different datasets)", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_yrand.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()


if __name__ == '__main__':
    main()