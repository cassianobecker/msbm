# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys
import pickle
import numpy as np
sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf as varinf


def main():

    file_list = np.array(sorted(os.listdir('data')))
    #We exclude those already trained (for debugging)
    ignore_list = [stri.replace("model_","") for stri in sorted(os.listdir('models'))]
    fil = [(name not in ignore_list) for name in file_list]
    file_list = file_list[fil]

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
        par = dict()
        hyper['M'] = data['M']
        hyper['Q'] = data['Q']

        par['MAX_ITER'] = 1000
        par['TOL_ELBO'] = 1.e-13
        par['ALG'] = 'cavi'
        par['kappas'] = np.ones(par['MAX_ITER'])
        
        #Best of 4
        candidate_moms = []
        candidate_elbos = []
        candidate_score = []
        for r in range(4):
            mom = im.init_moments(data, hyper, seed= r)
            results_mom, elbo_seq = varinf.infer(data, prior, hyper, mom, par)
            candidate_moms.append(results_mom)
            candidate_elbos.append(elbo_seq)
            candidate_score.append(elbo_seq['all'][-1])
        
        results_mom = candidate_moms[np.argmax(candidate_score)]
        elbo_seq = candidate_elbos[np.argmax(candidate_score)]

        print('Saving file to {:s} ... '.format('models/model_' + data_file))
        out_file_url = os.path.join('models', 'model_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))


if __name__ == '__main__':
    main()