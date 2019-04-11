# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np
import pdb

sys.path.insert(0, '../..')
import util as ut
import varinf
import updates_msbm_vi as upd


def main():
    file_list = sorted(os.listdir('data'))
    #We obtain statistics for all of the models in the form of a single table
    #were we register N, CH-Div, Q, and avg. rand index or avg entropy
    ch_list = []
    entro_list = []
    ari_list = []
    N_list = []
    Q_list = []

    for data_file in file_list:
        data = ut.load_data('data/' + data_file)
        model_url = os.path.join('models', 'model_' + data_file)
        results_mom, _ = ut.load_results(model_url)
        # z entropies
        entro_list.append(np.mean(ut.get_entropy_Z(results_mom)))
        # Adjusted rand index
        ari_list.append(ut.adj_rand_Z(results_mom, data))
        # And now from data we obtain the N, Q and CHdiv
        ch_list.append(data['SNR'])
        N_list.append(data['N'])
        Q_list.append(data['Q'])

    stats_url = os.path.join('stats','stats_detectability.pickle')    
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {
        'CH_div': ch_list,
        'ari_Z': ari_list,
        'N': N_list,
        'Q': Q_list,
        'a_entro': entro_list,
        }
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
