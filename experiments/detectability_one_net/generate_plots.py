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
    stats_url = os.path.join('stats', 'stats_' + 'detectability.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #box plot CH-div vs Rand Index
    #We create a list with the 8 boxes
    data1 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.55)&(chd<0.65) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])   
    data1 = data1[fil&fil2].flatten()

    data2 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.65)&(chd<0.75) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])
    data2 = data2[fil&fil2].flatten()

    data3 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.75)&(chd< 0.85) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])
    data3 = data3[fil&fil2].flatten()

    data4 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.85)&(chd<0.95) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])
    data4 = data4[fil&fil2].flatten()

    data5 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>0.95)&(chd<1.05) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])    
    data5 = data5[fil&fil2].flatten()

    data6 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>1.05)&(chd<1.15) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])    
    data6 = data6[fil&fil2].flatten()

    data7 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>1.15)&(chd<1.25) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])    
    data7 = data7[fil&fil2].flatten()

    data8 = np.array(statistics['ari_Z'])
    fil = np.array([(chd>1.25)&(chd<1.35) for chd in statistics['CH_div']])
    fil2 = np.array([n == 250 for n in statistics['N']])    
    data8 = data8[fil&fil2].flatten()

    data = [data1, data2, data3, data4, data5, data6, data7, data8]
    plt.boxplot(data)
    plt.xticks(range(1,9),[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], fontsize=10)
    plt.xlabel('CH Divergence')
    plt.ylabel('Adj. Rand Index')
    plt.title("CH-div vs Average Rand Index", fontsize= 16)
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_ch.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()

    #we make a boxplot for N vs ari_Z
    data1 = np.array(statistics['ari_Z'])
    fil = np.array([n == 75 for n in statistics['N']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    data1 = data1[fil&fil2].flatten()

    data2 = np.array(statistics['ari_Z'])
    fil = np.array([n == 151 for n in statistics['N']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    data2 = data2[fil&fil2].flatten()

    data3 = np.array(statistics['ari_Z'])
    fil = np.array([n == 300 for n in statistics['N']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    data3 = data3[fil&fil2].flatten()
    
    data = [data1, data2, data3]
    plt.boxplot(data)
    plt.title("N vs Average Rand Index at Threshold", fontsize= 16)
    plt.xticks(range(1,4),[75, 151, 300], fontsize=10)
    plt.xlabel('Nodes')
    plt.ylabel('Adj. Rand Index')
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_N.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()

    #we make a boxplot for Q vs ari_Z
    data1 = np.array(statistics['ari_Z'])
    fil = np.array([q == 2 for q in statistics['Q']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    fil3 = np.array([n == 149 for n in statistics['N']])
    data1 = data1[fil&fil2&fil3].flatten()

    data2 = np.array(statistics['ari_Z'])
    fil = np.array([q == 3 for q in statistics['Q']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    fil3 = np.array([n == 149 for n in statistics['N']])
    data2 = data2[fil&fil2&fil3].flatten()

    data3 = np.array(statistics['ari_Z'])
    fil = np.array([q == 4 for q in statistics['Q']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    fil3 = np.array([n == 149 for n in statistics['N']])
    data3 = data3[fil&fil2&fil3].flatten()

    data4 = np.array(statistics['ari_Z'])
    fil = np.array([q == 5 for q in statistics['Q']])
    fil2= np.array([(chd>1.00)&(chd<1.10) for chd in statistics['CH_div']])
    fil3 = np.array([n == 149 for n in statistics['N']])
    data4 = data4[fil&fil2&fil3].flatten()
    
    data = [data1, data2, data3, data4]
    plt.boxplot(data)
    plt.title("Q vs Average Rand Index at Threshold", fontsize= 16)
    plt.xticks(range(1,5),[2, 3, 4, 5], fontsize=12)
    plt.xlabel('Communities')
    plt.ylabel('Adj. Rand Index')
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_Q.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    sys.exit()

if __name__ == '__main__':
    main()