import numpy as np
import pdb
import numpy.random as npr
import networkx as nx

def init_moments(data, hyper):

    mom = dict()
    npr.seed(1)
    mode = 'random'
    mom['ALPHA'] = init_ALPHA(data, hyper, mode)
    mom['BETA'] = init_BETA(data, hyper, mode)
    mom['NU'] = init_NU(data, hyper, mode)
    mom['ZETA'] = init_ZETA(data, hyper, mode)

    mode = 'some_truth'
    mom['MU'] = init_MU(data, hyper, mode)
    mom['LOG_MU'] = np.log(mom['MU'])

    mode = 'distance_sparse'
    mom['TAU'] = init_TAU(data, hyper, mode)
    mom['LOG_TAU'] = np.log(mom['TAU'])

    return mom


def init_NU(data, hyper, mode='random'):

    if mode == 'random':
        NU = npr.rand(hyper['M'], hyper['Q'])
    if mode == 'uniform':
        NU = np.ones((hyper['M'], hyper['Q'])) / hyper['Q']

    return NU


def init_ZETA(data, hyper, mode='random'):

    if mode == 'random':
        ZETA = npr.rand(hyper['M'])
    if mode == 'uniform':
        ZETA = np.ones(hyper['M'])/hyper['M']

    return ZETA


def init_ALPHA(data, hyper, mode='random'):

    if mode == 'random':
        ALPHA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if mode == 'uniform':
        ALPHA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return ALPHA


def init_BETA(data, hyper, mode='random'):

    if mode == 'random':
        BETA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if mode == 'uniform':
        BETA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return BETA


def init_MU(data, hyper, mode='random'):

    if mode == 'random':
        MU = npr.rand(data['K'], hyper['M'])
    if mode == 'uniform':
        MU = np.ones((data['K'], hyper['M'])) / hyper['M']
    if mode == 'some_truth' :
        MU = data['Y']
        lamb = 0
        inds = npr.choice(data['Y'].shape[0], int(lamb*data['Y'].shape[0]), replace=False)   
        MU[inds] = npr.rand(len(inds), hyper['M'])

    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)

    return MU


def init_TAU(data, hyper, mode='random'):

    if mode == 'random':
        TAU = npr.rand(data['K'], hyper['M'], data['N'], hyper['Q'])
    if mode == 'uniform':
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
    if mode == 'distance' or mode == 'distance_sparse':
        #Initialize
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
        for k in range(data['K']):
            G = nx.from_numpy_matrix(data['X'][k,:])
            #Could we get rid of the M loop?
            for m in range(hyper['M']):
                #select Q seeds at random
                seeds = npr.choice(data['N'], hyper['Q'], replace=False)
                for q in range(hyper['Q']):
                    dists = nx.shortest_path_length(G,source= seeds[q])
                    #handle disconnected nodes
                    missing = np.array([(j not in dists.keys()) for j in range(data['N'])])
                    missing = np.array(range(data['N']))[missing]
                    for j in missing:
                        dists[j] = 100
                    dists = [dists[key] for key in sorted(dists.keys())]
                    TAU[k,m,: , q] = np.exp2(-np.array(dists))
                if mode == 'distance_sparse':
                    for n in range(data['N']):
                        TAU[k, m, n ,:] = (TAU[k, m, n ,:] == np.max(TAU[k, m, n ,:])) + 0
    if mode == 'truth':
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
        for m in range(hyper['M']):
            TAU[:, m, :, :] = data['Z']

    for k in range(data['K']):
        for m in range(hyper['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    return TAU
