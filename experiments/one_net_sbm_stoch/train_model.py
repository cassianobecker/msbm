# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model.

import sys, os
import pickle
import pdb
import numpy as np
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
#import varinf_stoch1 as varinf
import varinf_iter_stoch as varinf


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
        hyper['init_TAU'] = 'some_truth'
        hyper['init_LAMB_TAU'] = 0

        # initialize moments
        mom = im.init_moments(data, hyper)

        par = dict()
        par['MAX_ITER'] = 1000
        par['kappas'] = np.ones(par['MAX_ITER'])
        par['TOL_ELBO'] = 1.e-10
        par['ALG'] = 'natgrad'
        #par['nat_step'] = 1 #For comparing against plain varinf natgrad
        par['nat_step_rate'] = 0.8

        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        print('Saving file to {:s} ... '.format('models/model_' + data_file))
        out_file_url = os.path.join('models', 'model_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))


if __name__ == '__main__':
    main()
