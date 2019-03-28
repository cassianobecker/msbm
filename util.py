import sklearn.metrics as skm
import pickle
import numpy as np
from scipy.special import entr

# ################# PERSISTENCE FUNCTIONS ##############
def load_data(data_file_url):

    print('\nLoading data from {:s} ... '.format(data_file_url))
    loaded = pickle.load(open(data_file_url, 'rb'), encoding='latin1')
    print('loaded.')
    #(TO DO) this should only return data
    return loaded['data']

def load_results(result_file_url):

    print('\nLoading model from {:s} ... '.format(result_file_url))
    loaded = pickle.load(open(result_file_url, 'rb'), encoding='latin1')
    print('loaded.')
    #(TO DO) this should only return data
    return loaded['results_mom'], loaded['elbo_seq']

# ################# AUXILIARY FUNCTIONS #################
def find_col(idc):

    col = [np.nonzero(idc[i, :])[0][0] for i in range(idc.shape[0])]

    return col


def adj_rand(tau, X):

    ari = skm.adjusted_rand_score(find_col(X), np.argmax(tau, axis=1))

    return ari


def adj_rand_Z(mom, data):

    ms = np.argmax(mom['MU'], axis=1)
    aris = [adj_rand(mom['TAU'][k, m, :], data['Z'][k, :]) for k, m in enumerate(ms)]

    return aris

#Entropy of "correct" set of Z
def get_entropy_Z(mom):

    ms = np.argmax(mom['MU'], axis=1)
    entro = [entr(mom['TAU'][k, m, :]).sum(axis=1) for k, m in enumerate(ms)]

    return entro