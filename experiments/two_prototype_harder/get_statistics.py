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
import numpy.random as npr


def main():

    #load data
    data_file_list = sorted(os.listdir('data'))

    model_list = sorted(os.listdir('models'))

    model_list_v2_nu_sp = sorted([e for e in model_list if e.startswith('v2_nu_sp_')])
    model_list_v2_u_sp = sorted([e for e in model_list if e.startswith('v2_u_sp_')])
    model_list_v1_u_sp_ = sorted([e for e in model_list if e.startswith('v1_u_sp_')])
    model_list_v1_u_nsp_ = sorted([e for e in model_list if e.startswith('v1_u_nsp_')])
    model_list_v2_u_nsp = sorted([e for e in model_list if e.startswith('v2_u_nsp_')])

    yrand_v2_nu_sp_= []
    yrand_v2_u_sp_ = []
    yrand_v1_u_sp_ = []
    yrand_v1_u_nsp_ = []
    yrand_v2_u_nsp_ = []

    for i in range(50): 
        data_file = data_file_list[i]
        file_url = os.path.join('data', data_file)
        data = ut.load_data(file_url)

        model_file = model_list_v2_nu_sp[i]
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_v2_nu_sp.append(ut.adj_rand(results_mom['MU'], data['Y']))

        model_file = model_list_v2_u_sp[i]
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_v2_u_sp.append(ut.adj_rand(results_mom['MU'], data['Y']))

        model_file = model_list_v1_u_sp[i]
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_v1_u_sp.append(ut.adj_rand(results_mom['MU'], data['Y']))

        model_file = model_list_v1_u_nsp[i]
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_v1_u_nsp.append(ut.adj_rand(results_mom['MU'], data['Y']))

        model_file = model_list_v2_u_nsp[i]
        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_v2_u_nsp.append(ut.adj_rand(results_mom['MU'], data['Y']))

    stats_url = os.path.join('stats','stats_hard_inits.pickle')    
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {
        'yrand_v2_nu_sp_':yrand_v2_nu_sp_,
        'yrand_v2_u_sp_':yrand_v2_u_sp_,
        'yrand_v1_u_sp_':yrand_v1_u_sp_,
        'yrand_v1_u_nsp_':yrand_v1_u_nsp_,
        'yrand_v2_u_nsp_':yrand_v2_u_nsp_,
        }
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
