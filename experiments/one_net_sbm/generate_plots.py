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
    stats_url = os.path.join('stats', 'stats_' + 'one_net.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #density plot of entropies
    data = pd.DataFrame( statistics['entro_list'][0] , columns = ['Z_entropy'])
    plot1 = ggplot(data, aes('Z_entropy')) + geom_density() + ggtitle("Density of Z entropies")
    plot1_file = os.path.join('plots', 'plot_' + 'entropies.pdf')
    plot1.save(plot1_file)
    #line plot of ELBO
    elbos = statistics['elbo_seq']
    data2 = pd.DataFrame( {'TIME': range(len(elbos)) ,'ELBO': elbos })
    plot2 = ggplot(data2, aes(x= 'TIME' , y= 'ELBO')) + geom_line() + ggtitle("Evolution of ELBO")
    plot2_file = os.path.join('plots', 'plot_' + 'elbos.pdf')
    plot2.save(plot2_file)
    plt.clf()
    #True vs Obtained Gamma, Pi in single rectangle:
    Q = statistics['actual_gamma'].shape[0]
    N = len(statistics['entro_list'][0])
    data3 = np.zeros((N, N))
    data4 = np.zeros((N, N))
    for i in range(N): #rows
        for j in range(N): #columns
            q_order = list(i/N >= np.cumsum(statistics['actual_gamma']))
            q = q_order.index(0)
            q_prime_order = list(i/N >= np.cumsum(statistics['resulting_gamma']))
            q_prime= q_prime_order.index(0)

            r_order = list(j/N >= np.cumsum(statistics['actual_gamma']))
            r_prime_order = list(j/N >= np.cumsum(statistics['resulting_gamma']))
            r = r_order.index(0)
            r_prime = r_prime_order.index(0)
            data3[i,j] = statistics['actual_pi'][q,r]
            data4[i,j] = statistics['resulting_pi'][q_prime, r_prime]
    plt.imshow(data3, origin="upper", norm=colors.PowerNorm(0.5), cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title(r"Real edge probabilities, $\alpha$", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'actual_pi.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()

    #and we plot the resulting ones
    plt.imshow(data4, origin="upper", norm=colors.PowerNorm(0.5), cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title("Predicted edge probabilities", fontsize= 16)
    plot_file = os.path.join('plots', 'plot_' + 'pred_pi.svg')
    plt.savefig(plot_file, format="svg")
    sys.exit()


if __name__ == '__main__':
    main()