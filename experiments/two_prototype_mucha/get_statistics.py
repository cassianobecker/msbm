import os, sys, inspect
import pickle
import numpy as np

sys.path.insert(0, '../..')
import util as ut
import varinf


def main():
    file_list = sorted(os.listdir('data'))
    # Need to read in original data + trained moments and elbos
    chd_list = []
    ari_list = []
    delta_elbo_list = []
    entropy_list = []
    for data_file in file_list:
        print("Reading data + moments from model: {}".format(data_file))
        # load data
        data, par = ut.load_data('data/' + data_file)
        model_url = os.path.join('models', 'model_' + data_file)
        results_mom, elbo_seq = ut.load_results(model_url)

        chd_list.append(par['SNR'])
        # Adjusted rand index
        mari_Z = np.mean(ut.adj_rand_Z(results_mom, data))
        ari_list.append(mari_Z)
        # Percentage elbo increase
        delta_elbo = (elbo_seq[-1] - elbo_seq[0]) / np.abs(elbo_seq[0])
        delta_elbo_list.append(delta_elbo)
        # Mean z entropy
        entro = np.mean(ut.get_entropy_Z(results_mom))
        entropy_list.append(entro)

    stats_url = os.path.join('stats', 'stats_' + 'consensus.pickle')
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {'chd_list': chd_list, 'ari_list': ari_list, 'delta_elbo_list': delta_elbo_list, 'entropy_list': entropy_list}
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()