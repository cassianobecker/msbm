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
    data_file = os.listdir('data')[0]
    # Need to read in original data + trained moments and elbos
    print("Reading data + moments from model: {}".format(data_file))
    # load data
    data = ut.load_data('data/' + data_file)
    model_url = os.path.join('models', 'model_' + data_file)
    results_mom, elbo_seq = ut.load_results(model_url)
    # z entropies
    entro_list = ut.get_entropy_Z(results_mom)
    # Adjusted rand index
    ari_Z = ut.adj_rand_Z(results_mom, data)
    # Gamma and Pi vs Real Gamma and Pi
    resulting_pi = upd.Pi_from_mom(results_mom)[0, :, :]
    resulting_gamma = upd.Gamma_from_mom(results_mom)[0]

    actual_pi = data['PI'][0, :, :]
    actual_gamma = data['GAMMA'][0, :]

    stats_url = os.path.join('stats', 'stats_' + 'one_net.pickle')
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {
        'entro_list': entro_list,
        'ari_Z': ari_Z,
        'elbo_seq': elbo_seq['all'],
        'resulting_pi': resulting_pi,
        'resulting_gamma': resulting_gamma,
        'actual_pi': actual_pi,
        'actual_gamma': actual_gamma,
        }
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
