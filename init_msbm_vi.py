import numpy as np
import pdb
import numpy.random as npr
import networkx as nx

def init_moments(data, hyper, seed = None):

    mom = dict()
    npr.seed(seed)

    if 'init_LAMB_TAU' not in hyper.keys():
        hyper['init_LAMB_TAU'] = 0
    if 'init_LAMB_MU' not in hyper.keys():
        hyper['init_LAMB_MU'] = 0

    if 'init_others' not in hyper.keys():
        hyper['init_others'] = 'uniform'
    mom['ALPHA'] = init_ALPHA(data, hyper)
    mom['BETA'] = init_BETA(data, hyper)
    mom['NU'] = init_NU(data, hyper)
    mom['ZETA'] = init_ZETA(data, hyper)

    if 'init_MU' not in hyper.keys():
        hyper['init_MU'] = 'random'
    mom['MU'] = init_MU(data, hyper)
    mom['LOG_MU'] = np.log(mom['MU'])

    if 'init_MU' not in hyper.keys():
        hyper['init_MU'] = 'distance_sparse'
    mom['TAU'] = init_TAU(data, hyper)
    mom['LOG_TAU'] = np.log(mom['TAU'])

    return mom


def init_NU(data, hyper):

    if hyper['init_others'] == 'random':
        NU = npr.rand(hyper['M'], hyper['Q'])
    if hyper['init_others'] == 'uniform':
        NU = np.ones((hyper['M'], hyper['Q'])) / hyper['Q']

    return NU


def init_ZETA(data, hyper):

    if hyper['init_others'] == 'random':
        ZETA = npr.rand(hyper['M'])
    if hyper['init_others'] == 'uniform':
        ZETA = np.ones(hyper['M'])/hyper['M']

    return ZETA


def init_ALPHA(data, hyper):

    if hyper['init_others'] == 'random':
        ALPHA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if hyper['init_others'] == 'uniform':
        ALPHA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return ALPHA


def init_BETA(data, hyper):

    if hyper['init_others'] == 'random':
        BETA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if hyper['init_others'] == 'uniform':
        BETA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return BETA


def init_MU(data, hyper):

    if hyper['init_MU'] == 'random':
        MU = npr.rand(data['K'], hyper['M'])
    if hyper['init_MU'] == 'uniform':
        MU = np.ones((data['K'], hyper['M'])) / hyper['M']
    if hyper['init_MU'] == 'some_truth' :
        MU = data['Y'].copy()
        inds = npr.choice(data['Y'].shape[0], int(hyper['init_LAMB_MU']*data['Y'].shape[0]), replace=False)   
        MU[inds, :] = 1.0 - MU[inds, :]

    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)

    return MU

def init_TAU_distance(data, hyper):

    TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
    for k in range(data['K']):
        G = nx.from_numpy_matrix(data['X'][k,:])
        #Could we get rid of the M loop?
        for m in range(hyper['M']):
            #select Q seeds at random, avoiding neighbors
            probs = np.repeat(1/data['N'], data['N'])
            for q in range(hyper['Q']):
                mult = npr.multinomial(1, probs, 1)[0]
                anchor = np.nonzero(mult)[0][0]
                #adjust probabilities
                probs[anchor] = 0
                probs[G.neighbors(anchor)] = probs[G.neighbors(anchor)]/10
                probs = probs/sum(probs)
                #compute distances
                dists = nx.shortest_path_length(G,source= anchor)
                #handle disconnected nodes
                missing = np.array([(j not in dists.keys()) for j in range(data['N'])])
                missing = np.array(range(data['N']))[missing]
                for j in missing:
                    dists[j] = 100
                dists = [dists[key] for key in sorted(dists.keys())]
                #Turn into affinities
                TAU[k,m,: , q] = np.exp2(-np.array(dists))
            if hyper['init_TAU'] == 'distance_sparse':
                for n in range(data['N']):
                    TAU[k, m, n ,:] = (TAU[k, m, n ,:] == np.max(TAU[k, m, n ,:])) + 0    
    return TAU

def init_TAU(data, hyper):

    if hyper['init_TAU'] == 'random':
        TAU = npr.rand(data['K'], hyper['M'], data['N'], hyper['Q'])
    elif hyper['init_TAU'] == 'uniform':
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
    elif hyper['init_TAU'] == 'distance' or hyper['init_TAU'] == 'distance_sparse':
        TAU = init_TAU_distance(data, hyper)
    elif hyper['init_TAU'] == 'some_truth':
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
        for m in range(hyper['M']):
            TAU[:, m, :, :] = data['Z'].copy()
            inds = npr.choice(data['N'], int(hyper['init_LAMB_TAU']*data['N']), replace=False)
            TAU[:, m, inds, :] = 1.0 - TAU[:, m, inds, :] #we flip those indices

    for k in range(data['K']):
        for m in range(hyper['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    return TAU
