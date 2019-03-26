# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
	stats_url = os.path.join('stats', 'stats_' + 'consensus.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    fig1 = plt.figure()
    plt.plot(statistics['chd_list'], statistics['ari_list'])

    plot_file = os.path.join('plots', 'plot_' + 'consensus')
    plt.savefig(, format="svg")
    sys.exit()


if __name__ == '__main__':
    main()