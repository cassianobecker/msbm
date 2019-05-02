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
    file_list = sorted(os.listdir('data'))
    numb = 0
    for data_file in file_list:
  
        numb += 1 
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

        hyper['init_TAU'] = 'sbm'
        hyper['init_MU'] = 'spectral'
        hyper['init_others'] = 'random'

        # Varinf1, Unshuffle, Sparse
        print("----------------------------------")
        print("Training MSBM for dataset: {}".format(numb))
        mom = im.init_moments(data, hyper, seed= numb, sparse = False, unshuffle = True)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'msbm_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        #Just Spectral
        hyper['init_TAU'] = 'random'
        hyper['init_MU'] = 'spectral'
        hyper['init_others'] = 'random'

if __name__ == '__main__':
    main()