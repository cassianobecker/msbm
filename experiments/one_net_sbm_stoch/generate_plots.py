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

    stats_url = os.path.join('stats', 'results1.pickle')
    print("generating plots from: {}".format(stats_url))
    results = pickle.load(open(stats_url, 'rb'), encoding='latin1')

    #box plot z rand index
    #We create a list with the 4 boxes
    data1 = np.array(results['ari']['plain']).flatten()
    data2 = np.array(results['ari']['natgrad']).flatten()
    data3 = np.array(results['ari']['node']).flatten()
    data4 = np.array(results['ari']['dnode']).flatten()
    data5 = np.array(results['ari']['strat']).flatten()

    data = [data1, data2, data3, data4, data5]
    plt.boxplot(data)
    plt.xticks(range(1,6),['cavi', 'natgrad', 'stoch:node', 'stoch:dnode', 'stoch:strat'], fontsize=8)
    plt.xlabel('Inference Algorithm')
    plt.ylabel('Adj. Rand Index of Z')
    plt.title("Single Net Experiment.\n(Truth initialization (40\% true))", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'ari.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()

    #box plot times
    #We create a list with the 4 boxes
    data1 = np.array(results['times']['plain']).flatten()
    data2 = np.array(results['times']['natgrad']).flatten()
    data3 = np.array(results['times']['node']).flatten()
    data4 = np.array(results['times']['dnode']).flatten()
    data5 = np.array(results['times']['strat']).flatten()

    data = [data1, data2, data3, data4, data5]
    plt.boxplot(data)
    plt.xticks(range(1,6),['cavi', 'natgrad', 'stoch:node', 'stoch:dnode', 'stoch:strat'], fontsize=8)
    plt.xlabel('Inference Algorithm')
    plt.ylabel('Total Runtime')
    plt.title("Single Net Experiment.\n(Truth initialization (40\% true))", fontsize= 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'times.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()

    #Grouped boxplots by CH divergence
    # set width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = [np.mean(results['ari']['plain'])[np.arange(0,10)],
             np.mean(results['ari']['plain'])[np.arange(10,20)],
             np.mean(results['ari']['plain'])[np.arange(20,30)],
             np.mean(results['ari']['plain'])[np.arange(30,40)],
             np.mean(results['ari']['plain'])[np.arange(40,50)]]
    bars2 = [np.mean(results['ari']['natgrad'])[np.arange(0,10)],
             np.mean(results['ari']['natgrad'])[np.arange(10,20)],
             np.mean(results['ari']['natgrad'])[np.arange(20,30)],
             np.mean(results['ari']['natgrad'])[np.arange(30,40)],
             np.mean(results['ari']['natgrad'])[np.arange(40,50)]]
    bars3 = [np.mean(results['ari']['node'])[np.arange(0,10)],
             np.mean(results['ari']['node'])[np.arange(10,20)],
             np.mean(results['ari']['node'])[np.arange(20,30)],
             np.mean(results['ari']['node'])[np.arange(30,40)],
             np.mean(results['ari']['node'])[np.arange(40,50)]]
    bars4 = [np.mean(results['ari']['dnode'])[np.arange(0,10)],
             np.mean(results['ari']['dnode'])[np.arange(10,20)],
             np.mean(results['ari']['dnode'])[np.arange(20,30)],
             np.mean(results['ari']['dnode'])[np.arange(30,40)],
             np.mean(results['ari']['dnode'])[np.arange(40,50)]]
    bars5 = [np.mean(results['ari']['strat'])[np.arange(0,10)],
             np.mean(results['ari']['strat'])[np.arange(10,20)],
             np.mean(results['ari']['strat'])[np.arange(20,30)],
             np.mean(results['ari']['strat'])[np.arange(30,40)],
             np.mean(results['ari']['strat'])[np.arange(40,50)]]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    # Make the plot
    plt.bar(r1, bars1, color='#73b366', width=barWidth, edgecolor='white', label='cavi')
    plt.bar(r2, bars2, color='#48c52e', width=barWidth, edgecolor='white', label='natgrad')
    plt.bar(r3, bars3, color='#267416', width=barWidth, edgecolor='white', label='s_node')
    plt.bar(r4, bars4, color='#12360b', width=barWidth, edgecolor='white', label='s_dnode')
    plt.bar(r5, bars5, color='#072a00', width=barWidth, edgecolor='white', label='s_strat')

    # Add xticks on the middle of the group bars
    plt.ylabel('Average Rand Index for Z')
    plt.xlabel('Detectability', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['1.05', '1.20', '1.40', '1.60', '1.83'])

    # Create legend & Show graphic
    plt.legend()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_file = os.path.join('plots', 'plot_' + 'ari_detect.svg')
    plt.savefig(plot_file, format="svg")
    plt.clf()
    plt.close()

    sys.exit()

if __name__ == '__main__':
    main()