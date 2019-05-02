import os, sys, inspect
import pickle
import numpy as np
import pdb

sys.path.insert(0, '../..')
import util as ut
import varinf
import generate_msbm as gn

def main():
    file_list = sorted(os.listdir('data'))
    # Need to read in original data + trained moments and elbos

    ari_Y_list = []
    ari_spectral_list = []
    mari_Z_list = []
    chd_2_list = []
    pi_dist_list = []
    CH_dist_list = []
    for data_file in file_list:
        print("Reading data + moments from model: {}".format(data_file))
        # load data
        data = ut.load_data('data/' + data_file)
        model_file = data_file.replace('two_prot','msbm')
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)

        
        # Adjusted rand index
        ari_Y = ut.adj_rand(results_mom['MU'], data['Y'])
        ari_Y_list.append(ari_Y)
        # Adjusted rand index for Z
        mari_Z = np.mean(ut.adj_rand_Z(results_mom, data))
        mari_Z_list.append(mari_Z)
        # Rand Index of spectral method
        model_file = data_file.replace('two_prot','spectral')
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        ari_spectral= ut.adj_rand(results_mom['MU'], data['Y'])
        ari_spectral_list.append(ari_spectral) 

        #####
        # X- Axis
        #####
        #Detectability of second prototype
        chd_2_list.append(data['SNR'])
        ###
        #pi_gamma_dist after unshuffling
        #prototype 0
        gamma_0 = data['GAMMA'][0]
        pi_0 = data['PI'][0]
        #unshuffle
        PQ_0 = np.matmul(np.diag(gamma_0), pi_0)
        com_degs_0 = np.sum(PQ_0 * np.log(data['N']), axis=0)
        orders_0 = np.argsort(com_degs_0)
        gamma_0 = gamma_0[orders_0]
        pi_0 = pi_0[orders_0, :]
        pi_0 = pi_0[:, orders_0]        
        #prototype 1
        gamma_1 = data['GAMMA'][1]
        pi_1 = data['PI'][1]
        #unshuffle
        PQ_1 = np.matmul(np.diag(gamma_1), pi_1)
        com_degs_1 = np.sum(PQ_1 * np.log(data['N']), axis=0)
        orders_1 = np.argsort(com_degs_1)
        gamma_1 = gamma_1[orders_1]
        pi_1 = pi_1[orders_1, :]
        pi_1 = pi_1[:, orders_1]       

        pi_dist = np.sum( (gamma_0 - gamma_1)**2) + np.sum( (pi_0 - pi_1)**2)
        pi_dist_list.append(pi_dist)

        # Max Min CH-Div criterion
        ch_dist = gn.get_CH_dist(gamma_0, pi_0, gamma_1, pi_1, n = data['N'])
        CH_dist_list.append(ch_dist)

    pdb.set_trace()
    stats_url = os.path.join('stats', 'stats_' + 'mucha.pickle')
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {
        'ari_Y_list':ari_Y_list,
        'ari_spectral_list':ari_spectral_list,
        'mari_Z_list':mari_Z_list,
        'chd_2_list':chd_2_list,
        'pi_dist_list':pi_dist_list,
        'CH_dist_list':CH_dist_list,
        }
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
