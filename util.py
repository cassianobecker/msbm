import sklearn.metrics as skm
import pickle
import random 
import pdb
import networkx as nx
import numpy as np
from bisect import *
import heapq as hp
import scipy.sparse as sps
from scipy.special import entr

# ################# PERSISTENCE FUNCTIONS ##############
def load_data(data_file_url):

    print('\nLoading data from {:s} ... '.format(data_file_url))
    loaded = pickle.load(open(data_file_url, 'rb'), encoding='latin1')
    print('loaded.')
    #We add non-edges to the data upon loading (this was the fastest way)
    data = loaded['data']
    data['NON_X'] = 1.0 - data['X']
    for i in range(data['X'].shape[1]):
        data['NON_X'][:, i, i] = 0

    return data

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

#For doing node stratified sampling in stochastic varinf
def gen_stratified_sets(X, m):
    #generate a long list of size K*N, each element is a list of size m+1
    S = []
    for k in X.shape[0]:
        for i in range(X.shape[1]):
            l = []
            #put all the neighbors of X
            l.append(np.where(X[k, i, :] != 1)[0])
            #put m groups of non-edges that partition that set
            non_edges = np.where(X[k, i, :] == 0)[0]
            memberships = npr.choice(range(m), len(non_edges), True)
            for z in range(m):
                l.append(non_edges[np.where(memberships == z)])
            S.append(l)
            
    return S

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


def get_entropy_Y(mom):
    
    entro = [np.sum(entr(mom['MU'][k, :])) for k in range(mom['MU'].shape[0])]

    return entro

def get_laplacian(A):
    D = np.sum(A, axis=0)
    L = np.diag(D**(-1/2))@A@np.diag(D**(-1/2))
    return L


def get_NBT(X):

    #We get a list of the directed edges
    if sps.issparse(X):
        G = nx.from_scipy_sparse_matrix(X)
    else:
        G = nx.from_numpy_matrix(X)

    G = G.to_directed()
    dedges = G.edges()
    n = X.shape[0]
    m = len(dedges)

    #Some objects for efficient lookup
    dedges_src = [d[0] for d in dedges]
    dedges_trg = [d[1] for d in dedges]
    cutoffs_src = [bisect_left(dedges_src, x) for x in range(n)]
    cutoffs_src = np.append(cutoffs_src, m)

    NBT = nx.empty_graph(m).to_directed()    
    for i in range(m):        
        #Look-up the indices of dedges starting with i_trg and not ending in i
        i_src = dedges[i][0]
        i_trg = dedges[i][1]
        j_list = [j for j in range(cutoffs_src[i_trg], cutoffs_src[i_trg+1]) if dedges_trg[j] != i]
        for j in j_list:
            NBT.add_edge(i, j)

    return nx.to_scipy_sparse_matrix(NBT).asfptype(), dedges


def get_spectral_moms(X, orders):
    
    if not isinstance(orders, list):
        orders = [orders]
    spectral_moms = np.ones((X.shape[0], np.max(orders)+1))

    for k in range(X.shape[0]):
        A = np.identity(X.shape[1])
        for p in range(np.max(orders) + 1):
            if p > 0:
                A = np.matmul(A, X[k,:,:])
            spectral_moms[k, p] = np.trace(A)

    return (1/X.shape[1])*spectral_moms[:, orders]

def get_Y_distances( sp_moms ):
    net = np.zeros( (sp_moms.shape[0], sp_moms.shape[0]))
    #we standardize the attributes to take distances
    for p in range(sp_moms.shape[1]):
        sp_moms[:,p] = (sp_moms[:,p] - np.mean(sp_moms[:,p]))/np.std(sp_moms[:,p])

    #populate the matrix 
    for i in range(net.shape[0]):
        for j in range(i+1, net.shape[0]):
            net[i,j] = np.sum((sp_moms[i,:] - sp_moms[j,:])**2)

    net = net + np.transpose(net)
    return net

def peek_mom_TAU(mom, seed = None):
    """
    A peek at the TAU of 5 nodes selected at random from one 
    of the networks. The TAU set chosen is that of the likeliest
    prototype. 
    Parameters
    ----------
    mom : array
        array with all the moments
    seed: int
        a random seed to always obtain the same 5 nodes
    """
    print("Peek from Tau moments:")
    random.seed(seed)
    #select a random network 
    k = random.randint(0, mom['MU'].shape[0]-1)
    #assign prototype
    m = np.argmax(mom['MU'][k,:])
    #select the nodes at random
    node_ids = [random.randint(0,mom['TAU'].shape[2]-1) for i in range(5)]

    return mom['TAU'][k,m,node_ids,:]

