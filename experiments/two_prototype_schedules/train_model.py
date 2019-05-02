# We train the model with varying levels of difficulty starting from easiest (perfect initialization)
import os, sys
import pickle
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf1 as varinf
import numpy as np
import pdb

def main():

    #load data
    data_file = sorted(os.listdir('data'))[0]
    file_url = os.path.join('data', data_file)
    data = ut.load_data(file_url)

    prior = dict()
    prior['ALPHA_0'] = 0.5
    prior['BETA_0'] = 0.5
    prior['NU_0'] = 0.5
    prior['ZETA_0'] = 0.5

    hyper = dict()
    par = dict()
    par['MAX_ITER'] = 1000
    par['kappas'] = np.ones(par['MAX_ITER'])
    par['TOL_ELBO'] = 1.e-13
    par['ALG'] = 'cavi'

    hyper['M'] = data['M']
    hyper['Q'] = data['Q']

    #Noise for all the noisy initializations
    lamb = 0.3

    for numb in range(50):
        print("----------------------------------")
        print("Initialization: True Tau, Rand Mu")

        # initialize moments
        hyper['init_MU'] = 'random_sparse'
        hyper['init_LAMB_MU'] = 0
        hyper['init_TAU'] = 'some_truth'
        hyper['init_LAMB_TAU'] = 0
        hyper['init_others'] = 'random'

        mom = im.init_moments(data, hyper, seed = numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_tru_rand_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: Noisy Tau, True Mu")

        hyper['init_MU'] = 'some_truth'
        hyper['init_TAU'] = 'some_truth'
        hyper['init_LAMB_TAU'] = lamb

        mom = im.init_moments(data, hyper, seed = numb)        
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)


        outfile = 'model_noisy_tru_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: Tru Tau, Noisy Mu")

        hyper['init_LAMB_TAU'] = 0
        hyper['init_LAMB_MU'] = lamb 
        hyper['init_TAU'] = 'some_truth'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_tru_noisy_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: dist. Tau, Noisy Mu")

        hyper['init_LAMB_MU'] = lamb 
        hyper['init_TAU'] = 'distance_sparse'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_dist_noisy_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

if __name__ == '__main__':
    main()