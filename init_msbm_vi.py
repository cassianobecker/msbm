import numpy as np
import pdb
import numpy.random as npr
import networkx as nx
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import varinf2 as varinf
from util import *

def init_moments(data, hyper, seed = None, sparse = True, unshuffle = False):

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
        hyper['init_MU'] = 'distance'
    mom['MU'] = init_MU(data, hyper, seed)
    mom['LOG_MU'] = np.log(mom['MU'])

    if 'init_TAU' not in hyper.keys():
        hyper['init_TAU'] = 'spectral'
    mom['TAU'] = init_TAU(data, hyper, seed)
    if unshuffle:
        mom['TAU'] = unshuffle_TAU(data, hyper, mom['TAU'])
    mom['LOG_TAU'] = np.log(mom['TAU'])
    if sparse:
        for k in range(data['K']):
            mom['MU'][k, :] = (mom['MU'][k, :] == np.max(mom['MU'][k, :])) + 0 
            for m in range(hyper['M']):
                for n in range(data['N']):
                    mom['TAU'][k, m, n ,:] = (mom['TAU'][k, m, n ,:] == np.max(mom['TAU'][k, m, n ,:])) + 0  

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


def init_MU(data, hyper, seed):
    if hyper['M'] == 1:
        return np.ones((data['K'], hyper['M']))

    if hyper['init_MU'] == 'random':
        MU = npr.rand(data['K'], hyper['M'])
    elif hyper['init_MU'] == 'uniform':
        MU = np.ones((data['K'], hyper['M'])) / hyper['M']
    elif hyper['init_MU'] == 'distance':
        MU = init_MU_distance(data, hyper)
    elif hyper['init_MU'] == 'spectral':
        MU = init_MU_spectral(data, hyper, seed)
    elif hyper['init_MU'] == 'some_truth' :
        MU = data['Y'].copy()
        inds = npr.choice(data['Y'].shape[0], int(hyper['init_LAMB_MU']*data['Y'].shape[0]), replace=False)   
        MU[inds, :] = 1.0 - MU[inds, :]

    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)

    return MU

def init_MU_distance(data, hyper):
    #Compute moments
    MU = np.ones((data['K'], hyper['M']))
    sp_moms = get_spectral_moms(data['X'], orders = [2, 4, 6, 8])
    #Find unweighted distance matrix
    A = get_Y_distances( sp_moms )
    #Distance subroutine
    G = nx.from_numpy_matrix(A)

    #Select M seeds and initialize with exponential decay to the distance
    probs = np.repeat(1/data['K'], data['K'])
    #We don't want to select nodes of very low degree for this.
    DEGS = np.sum(A, axis = 0)
    probs = probs*(1 - np.exp(-DEGS/np.mean(DEGS)))
    probs = probs/np.sum(probs)

    for m in range(hyper['M']):
        mult = npr.multinomial(1, probs, 1)[0]
        anchor = np.nonzero(mult)[0][0]
        #adjust probabilities
        probs[anchor] = 0
        #compute distances
        dists = A[anchor, :]
        #Turn into affinities
        MU[:, m] = np.exp(-np.array(dists)/np.mean(dists))
        probs = probs*(1-np.exp(-np.array(dists)**2))
        probs = probs/sum(probs)
      
    return MU

def init_MU_spectral2(data, hyper, seed):
    #Compute moments
    MU = np.zeros((data['K'], hyper['M']))
    sp_moms = get_spectral_moms(data['X'], orders = [2, 4, 6, 8])
    #Find unweighted distance matrix
    A = get_Y_distances( sp_moms )
    #We turn into an affinity matrix
    A = np.exp( - A/np.mean(np.max(A)) )
    A = A - np.diag(np.diag(A))
    #We compute the normalized Laplacian
    D = np.sum(A, axis=0)
    L = np.diag(D) - A
    L = np.diag(D**(-1/2))@L@np.diag(D**(-1/2));
    #Get first M eigenvectors
    _, vecs = eigs(L, k= hyper['M'], which = 'SM')
    sp_emb = np.real(vecs[:, np.arange(1,hyper['M'])])

    #Perform k-means++
    kmeans = KMeans(n_clusters= hyper['M'], random_state= seed).fit(sp_emb)
    coms = kmeans.labels_
    for k in range(data['K']):
        com = coms[k]
        MU[k, com] += 1

    return MU

def init_MU_spectral(data, hyper, seed):
    #Compute moments
    MU = np.ones((data['K'], hyper['M']))/(10**12)
    sp_moms = get_spectral_moms(data['X'], orders = [2, 3, 4, 5, 6])
    #Find unweighted distance matrix
    A = get_Y_distances( sp_moms )
    #We turn into a binary affinity matrix
    A = A + 2*np.diag(np.repeat(np.max(np.max(A)), data['K']) )
    thresh = np.quantile( A.ravel(), 40/(2*data['N']) )
    A = (A <= thresh) + 0
    #We use the Non-backtracking operator for clustering
    NBT, dedges = get_NBT(A)

    #Get first M eigenvectors
    _, vecs = eigs(NBT, k= hyper['M'], which = 'LM')
    sp_emb = np.real(vecs[:, np.arange(1,hyper['M'])])
    #Perform k-means++
    kmeans = KMeans(n_clusters= hyper['M'], random_state= seed).fit(sp_emb)
    dedge_coms = kmeans.labels_
    points = [d[1] for d in dedges]

    for d in range(len(dedges)):
        com = dedge_coms[d]
        p = points[d]
        MU[p, com] += 1

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
  
    return TAU

def init_TAU_sbm(data, hyper, seed):
    #We initialize the TAU with distance based and then train a single sbm
    TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
    hyper_t = dict()
    par_t = dict()

    hyper_t['init_TAU'] = 'spectral'
    hyper_t['init_others'] = 'random'

    par_t['MAX_ITER'] = 200
    par_t['kappas'] = np.ones(par_t['MAX_ITER'])
    par_t['TOL_ELBO'] = 1.e-10
    par_t['ALG'] = 'cavi'

    prior_t = dict()
    prior_t['ALPHA_0'] = 0.5
    prior_t['BETA_0'] = 0.5
    prior_t['NU_0'] = 0.5
    prior_t['ZETA_0'] = 0.5    

    hyper_t['Q'] = hyper['Q']
    #We modify the data to only contain one network, and expect 1 prototype
    hyper_t['M'] = 1
    for k in range(data['K']):
        data_k = data.copy()
        data_k['K'] = 1
        data_k['X'] = data_k['X'][[k],:,:]
        data_k['NON_X'] = data_k['NON_X'][[k],:,:]
        data_k['Y'] = data_k['Y'][[k],:]
        data_k['Z'] = data_k['Z'][[k],:]

        mom_k = init_moments(data_k, hyper_t, seed)
        results_mom_k, elbo_seq = varinf.infer(data_k, prior_t, hyper_t, mom_k, par_t, False)

        for m in range(data['M']): #populate all with the same result
            TAU[k, m, :, :] = results_mom_k['TAU']

    return TAU

def init_TAU_spectral(data, hyper, seed):
    TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))/(10**12)
    for k in range(data['K']):
        #Obtain non-backtracking operator NBT
        NBT, dedges = get_NBT(data['X'][k, :, :])
        #Obtain the eigenvectors from 2 to Q+1
        _, vecs = eigs(NBT, k= hyper['Q'], which = 'LM')
        sp_emb = np.real(vecs[:, np.arange(1,hyper['Q'])])

        #Perform k-means++
        kmeans = KMeans(n_clusters= hyper['Q'], random_state= seed).fit(sp_emb)
        dedges_coms = kmeans.labels_
        points = [d[1] for d in dedges]
        #aggregate to node-level
        for d in range(len(dedges)):
            p = points[d]
            com = dedges_coms[d]
            for m in range(hyper['M']):
                TAU[k,m,p,com] += 1 
    return TAU


def init_TAU(data, hyper, seed):

    if hyper['init_TAU'] == 'random':
        TAU = npr.rand(data['K'], hyper['M'], data['N'], hyper['Q'])
    elif hyper['init_TAU'] == 'uniform':
        TAU = np.ones((data['K'], hyper['M'], data['N'], hyper['Q']))
    elif hyper['init_TAU'] == 'distance':
        TAU = init_TAU_distance(data, hyper)
    elif hyper['init_TAU'] == 'sbm':
        TAU = init_TAU_sbm(data, hyper, seed)
    elif hyper['init_TAU'] == 'spectral':
        TAU = init_TAU_spectral(data, hyper, seed)
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

def unshuffle_TAU( data, hyper, TAU ):
    #We do an estimate of Pi and Gamma and sort
    #according to expected degree per community
    for k in range(data['K']):
        gamma = np.sum(TAU[k,0, :, :], axis = 0)
        gamma = gamma/sum(gamma)

        str_sum = 'ij, iq, jr -> qr'
        alpha = np.einsum(str_sum, data['X'][k,:,:],
                                          TAU[k, 0, :, :], TAU[k, 0, :, :])
        beta = np.einsum(str_sum, data['NON_X'][k, :, :],
                                          TAU[k, 0, :, :], TAU[k, 0, :, :])
        pi = alpha/(alpha + beta)

        PQ = np.matmul(np.diag(gamma), pi)
        com_degs = np.sum(PQ * np.log(data['N']), axis=0)

        orders = np.argsort(com_degs)
        for m in range(hyper['M']):
            TAU[k, m, :, :] = np.transpose(TAU[k, m, :, orders])
    #Now we permute the columns of Tau accordingly
    return TAU