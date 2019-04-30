# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys
import pickle
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf
import numpy as np


def main():

    file_list = sorted(os.listdir('data'))

    for data_file in file_list:
        # load data
        file_url = os.path.join('data', data_file)
        data = ut.load_data(file_url)

        prior = dict()
        prior['ALPHA_0'] = 0.5
        prior['BETA_0'] = 0.5
        prior['NU_0'] = 0.5
        prior['ZETA_0'] = 0.5

        # assigning hyper-parameters from ground truth (cheating)
        hyper = dict()
        hyper['M'] = data['M']
        hyper['Q'] = data['Q']

        # initialize moments
        hyper['init_MU'] = 'some_truth'
        hyper['init_LAMB_MU'] = 0
        hyper['init_TAU'] = 'distance_sparse'
        hyper['init_LAMB_TAU'] = 0
        hyper['init_others'] = 'uniform'

        mom = im.init_moments(data, hyper)
        par = dict()
        par['MAX_ITER'] = 500
        par['TOL_ELBO'] = 1.e-13
        par['ALG'] = 'cavi'
        par['kappas'] = np.ones(par['MAX_ITER'])

        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        print('Saving file to {:s} ... '.format('models/model_' + data_file))
        out_file_url = os.path.join('models', 'model_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))


if __name__ == '__main__':
    main()