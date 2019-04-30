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
    stats_url = os.path.join('stats', 'stats_easiest_inits.pickle')
    print("generating plots from: {}".format(stats_url))
    statistics = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #box plot y rand index
    #We create a list with the 4 boxes
    data1 = np.array(statistics['yrand_tru_rand']).flatten()
    data2 = np.array(statistics['yrand_noisy_tru']).flatten()
    data3 = np.array(statistics['yrand_tru_noisy']).flatten()
    data4 = np.array(statistics['yrand_dist_noisy']).flatten()

    data = [data1, data2, data3, data4]
    plt.boxplot(data)
    plt.xticks(range(1,5),['tru;rand', 'noisy;tru', 'tru;noisy', 'dist;noisy'], fontsize=8)
    plt.xlabel('Initialization method')
    plt.ylabel('Adj. Rand Index of Y')
    plt.title("Rand Index for different Inits.\n(50 draws each, 0.3 noise)", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_yrand.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()

    #we make a boxplot for final elbo vs init
    data1 = np.array(statistics['elbo_tru_rand']).flatten()
    data2 = np.array(statistics['elbo_noisy_tru']).flatten()
    data3 = np.array(statistics['elbo_tru_noisy']).flatten()
    data4 = np.array(statistics['elbo_dist_noisy']).flatten()

    data = [data1, data2, data3, data4]
    plt.boxplot(data)
    plt.xticks(range(1,5),['tru;rand', 'noisy;tru', 'tru;noisy', 'dist;noisy'], fontsize=8)
    plt.xlabel('Initialization method')
    plt.ylabel('Final Elbo for MSBM')
    plt.title("Effect of Initialization on ELBO\n(50 draws each, 0.3 noise)", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    plot_file = os.path.join('plots', 'plot_' + 'boxplot_elbos.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()

    #we obtain a sample of the trained models for tru - noisy and get the avg. rand index and entropy
    #True vs Obtained Gamma, Pi in single rectangle:
    Q = statistics['gamma'].shape[1]
    N = statistics['N']

    for prototype in range(statistics['gamma'].shape[0]):
        data_true = np.zeros((N, N))
        data_pred = np.zeros((N, N))
        for i in range(N): #rows
            for j in range(N): #columns
                q_order = list(i/N >= np.cumsum(statistics['gamma'][prototype]))
                q = q_order.index(0)
                q_prime_order = list(i/N >= np.cumsum(statistics['final_gamma'][prototype]))
                q_prime= q_prime_order.index(0)

                r_order = list(j/N >= np.cumsum(statistics['gamma'][prototype]))
                r_prime_order = list(j/N >= np.cumsum(statistics['final_gamma'][prototype]))
                r = r_order.index(0)
                r_prime = r_prime_order.index(0)
                data_true[i,j] = statistics['pi'][prototype,q,r]
                data_pred[i,j] = statistics['final_pi'][prototype,q_prime, r_prime]

        plt.imshow(data_true, origin="upper", vmin= 0, vmax= 1, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title(r"Real ($\pi_{}, \gamma_{}$)".format(prototype, prototype), fontsize= 16)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        cum_gamma = np.concatenate(([0],np.cumsum(statistics['gamma'][prototype])))
        for i in range(Q):
            for j in range(Q):
                #Get coordinates of 'typical pixel'
                n_i = N*(cum_gamma[i] + cum_gamma[i+1])/2
                n_j = N*(cum_gamma[j] + cum_gamma[j+1])/2
                plt.text(n_i, n_j, round(data_true[int(n_i), int(n_j)],2),
                           ha="center", va="center", fontsize = 20)    

        plot_file = os.path.join('plots', 'plot_actual_pi_{}.svg'.format(prototype))
        plt.savefig(plot_file, format="svg")
        plt.clf()
        plt.close()
        #and we plot the resulting ones

        plt.imshow(data_pred, origin="upper", vmin= 0, vmax= 1, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title(r"Pred (dist, noise) $(\pi_{}, \gamma_{})$ --(Z ari: {}, Z entr: {})".format(
            prototype, prototype,
            round(statistics['mari_Z'],2),
            round(statistics['mentro_Z'],2),
            ), fontsize= 16)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        cum_gamma = np.concatenate(([0],np.cumsum(statistics['final_gamma'][prototype])))
        for i in range(Q):
            for j in range(Q):
                #Get coordinates of 'typical pixel'
                n_i = N*(cum_gamma[i] + cum_gamma[i+1])/2
                n_j = N*(cum_gamma[j] + cum_gamma[j+1])/2
                plt.text(n_i, n_j, round(data_pred[int(n_i), int(n_j)],2),
                           ha="center", va="center", fontsize = 20)    
        plot_file = os.path.join('plots', 'plot_pred_pi_{}.svg'.format(prototype))
        plt.savefig(plot_file, format="svg")
        plt.clf()
        plt.close()
    sys.exit()

if __name__ == '__main__':
    main()