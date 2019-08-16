# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model.

import sys, os
import pickle
import pdb
import numpy as np
import time
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf as varinf_plain
import varinf_stoch as varinf_st_node
import varinf_stoch1 as varinf_st_dnode
import varinf_iter_stoch as varinf_st_strat


def main():

    file_list = sorted(os.listdir('data'))

    #We want to run all the varinfs on all 50 of the datasets, but in a way amenable to 
    #pausing... So we can keep a list with the results which we just load and unload,
    #with a counter that depends on the number of elements in that list.
    prior = dict()
    prior['ALPHA_0'] = 0.5
    prior['BETA_0'] = 0.5
    prior['NU_0'] = 0.5
    prior['ZETA_0'] = 0.5

    par = dict()
    par['MAX_ITER'] = 4000
    par['kappas'] = np.ones(par['MAX_ITER'])
    par['TOL_ELBO'] = 1.e-8

    #Create the object that will store the times and ari of the experiment
    if not os.path.isfile('statistics/results.pickle'):
        results = {'ari' : {}, 'times' : {}}
        results['ari']['plain'] = []
        results['times']['plain'] = []
        results['ari']['natgrad'] = []
        results['times']['natgrad'] = []
        results['ari']['node'] = []
        results['times']['node'] = []
        results['ari']['dnode'] = []
        results['times']['dnode'] = []
        results['ari']['strat'] = []
        results['times']['strat'] = []

        print('Saving file to statistics/results.pickle')
        out_file_url = 'stats/results.pickle'
        pickle.dump(results, open(out_file_url, 'wb'))
    #Import results file to find initial index
    results = pickle.load(open('stats/results.pickle', 'rb'), encoding='latin1')
    current = len(results['ari']['plain'])

    for i in np.arange(current, 50):

        print('###########Training models for network {:02d}###############'.format(i))
        # load data
        data_file = 'net{:02d}.pickle'.format(i)
        file_url = os.path.join('data', data_file)
        data = ut.load_data(file_url)

        # assigning hyper-parameters from ground truth (cheating)
        hyper = dict()
        hyper['M'] = data['M']
        hyper['Q'] = data['Q']
        hyper['init_TAU'] = 'some_truth'
        hyper['init_LAMB_TAU'] = 0.60

        # initialize moments
        initial_mom = im.init_moments(data, hyper)

        par['ALG'] = 'cavi'

        #plain CAVI
        tic = time.clock()
        mom = initial_mom.copy()
        results_mom, _ = varinf_plain.infer(data, prior, hyper, mom, par)
        results['times']['plain'].append(time.clock() - tic)
        results['ari']['plain'].append(np.mean(ut.adj_rand_Z(results_mom, data)))

        par['ALG'] = 'natgrad'
        par['nat_step_rate'] = 0.85

        #deterministic natgrad
        tic = time.clock()
        mom = initial_mom.copy()
        results_mom, _ = varinf_plain.infer(data, prior, hyper, mom, par)
        results['times']['natgrad'].append(time.clock() - tic)
        results['ari']['natgrad'].append(np.mean(ut.adj_rand_Z(results_mom, data)))

        #node sampling stochastic varinf
        tic = time.clock()
        mom = initial_mom.copy()
        results_mom, _ = varinf_st_node.infer(data, prior, hyper, mom, par)
        results['times']['node'].append(time.clock() - tic)
        results['ari']['node'].append(np.mean(ut.adj_rand_Z(results_mom, data)))

        #double node sampling stochastic varinf
        tic = time.clock()
        mom = initial_mom.copy()
        results_mom, _ = varinf_st_dnode.infer(data, prior, hyper, mom, par)
        results['times']['dnode'].append(time.clock() - tic)
        results['ari']['dnode'].append(np.mean(ut.adj_rand_Z(results_mom, data)))

        #stratified double node sampling
        tic = time.clock()
        mom = initial_mom.copy()
        results_mom, _ = varinf_st_strat.infer(data, prior, hyper, mom, par)
        results['times']['strat'].append(time.clock() - tic)
        results['ari']['strat'].append(np.mean(ut.adj_rand_Z(results_mom, data)))

        print('Saving file to statistics/results.pickle')
        out_file_url = 'stats/results.pickle'
        pickle.dump(results, open(out_file_url, 'wb'))


if __name__ == '__main__':
    main()
