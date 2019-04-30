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
    data_file = sorted(os.listdir('data'))[0]
    file_url = os.path.join('data', data_file)
    data = ut.load_data(file_url)

    model_list = sorted(os.listdir('models'))
    #Obtain Y. Rand Index and Final Elbo list for each model
    yrand_tru_tru = []
    elbo_tru_tru = []
    yrand_noisy_tru = []
    elbo_noisy_tru = []
    yrand_tru_noisy = []
    elbo_tru_noisy = []
    yrand_dist_noisy = []
    elbo_dist_noisy = []
    #We subset to those of the form tru tru
    model_list_tru_tru = [e for e in model_list if e.startswith('model_tru_tru')]
    for model_file in model_list_tru_tru:

        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_tru_tru.append(ut.adj_rand(results_mom['MU'], data['Y']))
        # final elbo
        elbo_tru_tru.append(elbo_seq['all'][-1])

    model_list_noisy_tru = [e for e in model_list if e.startswith('model_noisy_tru')]
    for model_file in model_list_noisy_tru:

        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_noisy_tru.append(ut.adj_rand(results_mom['MU'], data['Y']))
        # final elbo
        elbo_noisy_tru.append(elbo_seq['all'][-1])

    model_list_tru_noisy = [e for e in model_list if e.startswith('model_tru_noisy')]
    for model_file in model_list_tru_noisy:

        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_tru_noisy.append(ut.adj_rand(results_mom['MU'], data['Y']))
        # final elbo
        elbo_tru_noisy.append(elbo_seq['all'][-1])

    model_list_dist_noisy = [e for e in model_list if e.startswith('model_dist_noisy')]
    for model_file in model_list_dist_noisy:

        model_url = os.path.join('models', model_file)
        results_mom, elbo_seq = ut.load_results(model_url)
        # y_rand
        yrand_dist_noisy.append(ut.adj_rand(results_mom['MU'], data['Y']))
        # final elbo
        elbo_dist_noisy.append(elbo_seq['all'][-1])

    # Gamma and Pi vs Real Gamma and Pi
    #Select a seed at random
    seed = npr.choice(range(len(model_list_dist_noisy)))

    model_file = model_list_dist_noisy[seed]
    model_url = os.path.join('models', model_file)
    results_mom, elbo_seq = ut.load_results(model_url)    

    final_pi = upd.Pi_from_mom(results_mom)
    final_gamma = upd.Gamma_from_mom(results_mom)

    pi = data['PI']
    gamma = data['GAMMA']

    mari_Z = np.mean(ut.adj_rand_Z(results_mom, data))
    mentro_Z = np.mean(ut.get_entropy_Z(results_mom))

    stats_url = os.path.join('stats','stats_easiest_inits.pickle')    
    print('Saving file to {:s} ... '.format(stats_url))
    stats_dict = {
        'yrand_tru_tru': yrand_tru_tru,
        'elbo_tru_tru': elbo_tru_tru,
        'yrand_noisy_tru': yrand_noisy_tru,
        'elbo_noisy_tru': elbo_noisy_tru,
        'yrand_tru_noisy': yrand_tru_noisy,
        'elbo_tru_noisy': elbo_tru_noisy,
        'yrand_dist_noisy': yrand_dist_noisy,
        'elbo_dist_noisy': elbo_dist_noisy,
        'final_pi': final_pi,
        'final_gamma': final_gamma,
        'pi': pi,
        'gamma': gamma,  
        'N': data['N'],
        'seed': seed,
        'mari_Z': mari_Z,
        'mentro_Z': mentro_Z,
        }
    pickle.dump(stats_dict, open(stats_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
