# We train the model with varying levels of difficulty starting from easiest (perfect initialization)
import os, sys
import pickle
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf2 as varinf2
import varinf1 as varinf1
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
        par['TOL_ELBO'] = 1.e-9
        par['ALG'] = 'cavi'

        hyper['M'] = data['M']
        hyper['Q'] = data['Q']

        hyper['init_TAU'] = 'spectral'
        hyper['init_MU'] = 'spectral'
        hyper['init_others'] = 'random'

        #Plain initialization varinf2, no unshuffle, sparse
        print("----------------------------------")
        print("Training varinf2, no unshuffle, sparse for: {}".format(data_file))
        mom = im.init_moments(data, hyper, seed= numb, sparse = True, unshuffle = False)
        results_mom, elbo_seq = varinf2.infer(data, prior, hyper, mom, par)

        outfile = os.path.join('models', 'v2_nu_sp_' + data_file)
        print('Saving file to {:s} ... '.format(outfile))
        out_file_url = os.path.join('models', 'v2_nu_sp_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        # varinf2, with unshuffle, sparse
        print("----------------------------------")
        print("Training varinf2, unshuffle, sparse for: {}".format(data_file))        
        mom = im.init_moments(data, hyper, seed= numb, sparse = True, unshuffle = True)
        results_mom, elbo_seq = varinf2.infer(data, prior, hyper, mom, par)

        outfile = os.path.join('models', 'v2_u_sp_' + data_file)
        print('Saving file to {:s} ... '.format(outfile))
        out_file_url = os.path.join('models', 'v2_u_sp_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        # varinf1, with unshuffle, sparse
        print("----------------------------------")
        print("Training varinf1, unshuffle, sparse for: {}".format(data_file))                
        mom = im.init_moments(data, hyper, seed= numb, sparse = True, unshuffle = True)
        results_mom, elbo_seq = varinf1.infer(data, prior, hyper, mom, par)

        outfile = os.path.join('models', 'v1_u_sp_' + data_file)
        print('Saving file to {:s} ... '.format(outfile))
        out_file_url = os.path.join('models', 'v1_u_sp_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        # varinf1, with unshuffle, not sparse
        print("----------------------------------")
        print("Training varinf1, unshuffle, not sparse for: {}".format(data_file))        
        mom = im.init_moments(data, hyper, seed= numb, sparse = False, unshuffle = True)
        results_mom, elbo_seq = varinf1.infer(data, prior, hyper, mom, par)

        outfile = os.path.join('models', 'v1_u_nsp_' + data_file)
        print('Saving file to {:s} ... '.format(outfile))
        out_file_url = os.path.join('models', 'v1_u_nsp_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        # varinf2, with unshuffle, not sparse
        print("----------------------------------")
        print("Training varinf2, unshuffle, not sparse for: {}".format(data_file))        
        mom = im.init_moments(data, hyper, seed= numb, sparse = False, unshuffle = True)
        results_mom, elbo_seq = varinf2.infer(data, prior, hyper, mom, par)

        outfile = os.path.join('models', 'v2_u_nsp_' + data_file)
        print('Saving file to {:s} ... '.format(outfile))
        out_file_url = os.path.join('models', 'v2_u_nsp_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

if __name__ == '__main__':
    main()