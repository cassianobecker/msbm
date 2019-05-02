# We train the model with varying levels of difficulty starting from easiest (perfect initialization)
import os, sys
import pickle
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf2 as varinf
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

    #Realistic Initializations Including:
    #- Random (Mu) *
    #- Distance Mu *
    #- Spectral (clustering) Mu (With spectral moments distance)

    #- Distance Tau *
    #- SBM Tau *
    #- Spectral Tau

    #WITH VARINF2
    for numb in range(5):
        print("----------------------------------")
        print("Initialization: SBM Tau, Rand Mu")

        hyper['init_MU'] = 'random'
        hyper['init_TAU'] = 'sbm'
        hyper['init_others'] = 'random'

        mom = im.init_moments(data, hyper, seed = numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_sbm_rand_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: SBM Tau, dist. Mu")

        hyper['init_MU'] = 'distance'
        hyper['init_TAU'] = 'sbm'
        hyper['init_others'] = 'random'

        mom = im.init_moments(data, hyper, seed = numb)        
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)


        outfile = 'model_sbm_dist_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: SBM Tau, spectral Mu")

        hyper['init_TAU'] = 'spectral'
        hyper['init_MU'] = 'spectral'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_sbm_spectral_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: dist. Tau, Spectral Mu")

        hyper['init_TAU'] = 'distance'
        hyper['init_MU'] = 'spectral'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_dist_spectral_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: Spectral Tau, Spectral Mu")

        hyper['init_TAU'] = 'spectral'
        hyper['init_MU'] = 'spectral'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_spectral_spectral_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

        print("----------------------------------")
        print("Initialization: Distance Tau, Distance Mu")

        hyper['init_TAU'] = 'distance'
        hyper['init_MU'] = 'distance'

        mom = im.init_moments(data, hyper, seed= numb)
        results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)

        outfile = 'model_dist_dist_{:02}.pickle'.format(numb)
        print('Saving file to {:s} ... '.format('models/' + outfile))
        out_file_url = os.path.join('models', outfile)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))

if __name__ == '__main__':
    main()