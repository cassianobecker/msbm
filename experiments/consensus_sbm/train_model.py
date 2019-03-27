# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os
import pickle
import sys

sys.path.insert(0, '../..')
import util as ut
import init_msbm_vi as im
import varinf


def main():
    file_list = sorted(os.listdir('data'))

    for data_file in file_list:
        # load data
        file_url = os.path.join('data', data_file)
        data, par = ut.load_data(file_url)
        # initialize moments
        # (TO DO) init_moments should get a "hyper" instead of a par
        # and it's responsibility of the user to provide it
        # hyper = {}/or get it from data
        mom, prior = im.init_moments(par)
        # set max iterations
        par['MAX_ITER'] = 50
        par['TOL_ELBO'] = 1.e-14
        par['M'] = data['M'] 
        par['Q'] = data['Q']
        # (TO DO) infer should need only algorithmic pars
        results_mom, elbo_seq = varinf.infer(mom, data, prior, par, 'cavi')
        print('Saving file to {:s} ... '.format('models/model_' + data_file))
        out_file_url = os.path.join('models', 'model_' + data_file)
        pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open(out_file_url, 'wb'))
    sys.exit()


if __name__ == '__main__':
    main()
