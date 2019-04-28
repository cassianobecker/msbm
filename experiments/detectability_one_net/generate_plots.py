# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys
import pickle
import pdb
import pandas as pd 
from ggplot import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def main():
    stats_url = os.path.join('stats', 'stats_' + 'detectability.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #box plot CH-div vs Rand Index
    #We create a list with the 5 boxes
    data1 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.45)&(chd<0.55) for chd in statistics['CH_div']])
    data1 = data1[fil]

    data2 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.75)&(chd<0.85) for chd in statistics['CH_div']])
    data2 = data2[fil]

    data3 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.95)&(chd<1.05) for chd in statistics['CH_div']])
    data3 = data3[fil]

    data4 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>1.15)&(chd<1.25) for chd in statistics['CH_div']])
    data4 = data4[fil]

    data5 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>1.45)&(chd<1.55) for chd in statistics['CH_div']])
    data5 = data5[fil]

    data = [data1.flatten(), data2.flatten(), data3.flatten(), data4.flatten(), data5.flatten()]
    plt.boxplot(data)
    plt.xticks(range(1,6),[0.5, 0.8, 1, 1.2, 1.5], fontsize=8)
    plt.xlabel('CH Divergence')
    plt.ylabel('Adj. Rand Index')
    plt.title("CH-div vs Average Rand Index", fontsize= 16)
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_ch.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    #we make a boxplot for N vs ari_Z
    data1 = np.array(statistics['ari_Z'])
    fil = np.array([n == 75 for n in statistics['N']])
    fil2= np.array([(chd>1.05)&(chd<1.15) for chd in statistics['CH_div']])
    data1 = data1[fil&fil2]

    data2 = np.array(statistics['ari_Z'])
    fil = np.array([n == 150 for n in statistics['N']])
    fil2= np.array([(chd>1.05)&(chd<1.15) for chd in statistics['CH_div']])
    data2 = data2[fil&fil2]

    data3 = np.array(statistics['ari_Z'])
    fil = np.array([n == 300 for n in statistics['N']])
    fil2= np.array([(chd>1.05)&(chd<1.15) for chd in statistics['CH_div']])
    data3 = data3[fil&fil2]
    
    data = [data1.flatten(), data2.flatten(), data3.flatten()]
    plt.boxplot(data)
    plt.title("N vs Average Rand Index at Threshold", fontsize= 16)
    plt.xticks(range(1,4),[75, 150, 300], fontsize=8)
    plt.xlabel('Nodes')
    plt.ylabel('Adj. Rand Index')
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_N.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    sys.exit()


if __name__ == '__main__':
    main()