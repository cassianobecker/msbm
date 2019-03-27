"""
Set of functions to generate synthetic data for experiments as in
the paper "Clustering network layers with the strata multilayer
stochastic block model"
"""
import numpy as np
import numpy.random as npr
import sys
import pickle
import time
import pdb
from random import randint
import scipy as sp
from scipy.stats import beta
from scipy import optimize
from scipy import sparse
import generate_msbm as gen

def sample_Y_balanced(M, K):
    """
    Sample the vector of cluster memberships, i.e. the type
    of each network
    Parameters
    ----------
    M : integer 
        the number of clusters or prototypes
    K : integer
        Number of networks
    """
    K_M = np.floor(K/M)
    #we sample the excedent at random
    Y = npr.multinomial(1,np.ones(M)/M, int(K - K_M*M))
    for m in range(M):
        probs = np.zeros(M)
        probs[m] += 1
        Y = np.concatenate((Y,npr.multinomial(1,probs, int(K_M))))
    npr.shuffle(Y)
    return Y.astype(float)

def sample_Z_balanced(Q, N):
    """
    Sample the vector of cluster memberships, i.e. the type
    of each network
    Parameters
    ----------
    Q : integer 
        the number of communities
    N : integer
        Number of nodes
    """
    N_Q = np.floor(N/Q)
    #we sample the excedent at random
    Z = npr.multinomial(1,np.ones(Q)/Q,int(N - N_Q*Q))
    for q in range(Q):
        probs = np.zeros(Q)
        probs[q] += 1
        Z = np.concatenate((Z,npr.multinomial(1, probs, int(N_Q))))
    npr.shuffle(Z)
    return Z.astype(float)

def create_multistrata(
    Q= 4, N = 128, K = 100, M=2,
    pii = [0.1836,0.1836], c = 16,
    path_data = 'data',
    fname = 'multistrata',
    verbose = False):
    """
    Sample multiple networks according to the MSBM model according to 
    the given parameters and a target Chernof-Hellinger divergence SNR.
    Parameters
    ----------
    Q : integer 
        number of communities in each network
    N : integer
        number of nodes in each network
    M : integer
        number of different strata
    K : integer
        number of layers 
    pii : double 1d-array
        pii is the whithin community connection probabilities
    c : double
        average degree of the network such that c = (n/Q)(pii + (Q-1)pij)
    path_data : string
        the folder where data is to be stored for the current experiment
    fname : string
        name of the pickle file for persistance
    verbose : boolean
        set to True to obtain details during the execution
    """
    if len(pii) != M:
        sys.exit("Pii vector is of dimension {} but M = {}".format(len(pii),M))
    if verbose == True:
        print('---- Creating multistrata model with N = {:d}, K = {:d}, Q = {:d} and M = {:d} ------'
          .format(N, K, Q, M))
    #Sample the (deterministic) cluster memberships	
    RHO = np.ones(M)/M
    Y = sample_Y_balanced(M, K)
    GAMMA = np.ones((M,Q))/Q

    #To get Pi first we obtain pij from c and pii
    PI = np.ones((M, Q, Q))
    #We create a vector that will store the C-H Divergence of the generated models
    SNR = np.zeros(M)
    for m in range(M):
        if verbose == True:
            print("GENERATING PROTOTYPE NUMBER: {:d}".format(m))
        pij = ((Q*c)/N - pii[m])/(Q-1)
        PI[m, :] = PI[m, :]*pij + np.diag( np.repeat(pii[m] - pij,Q))
        pi_constant = PI[m, :] * (N/np.log(N))
        gamma = GAMMA[m, :].reshape((1,Q))
        SNR[m], _, _ =  gen.getSNR(gamma,pi_constant)
        if verbose == True:
            print("Prototype {} has PI matrix:".format(m))
            print(PI[m, :])
            print("with C-H Divergence of: {:03f}".format(SNR[m]))

    Z = np.zeros((K, N, Q))
    X = np.zeros((K, N, N))
    for k in range(K):
        m = gen.find_row(Y[k, :])
        Z[k, :] = sample_Z_balanced(Q, N)
        X[k, :] = gen.sample_X_und(PI[m, :], Z[k, :])

    data = dict()

    data['Q'] = Q
    data['N'] = N
    data['M'] = M
    data['K'] = K
    data['pii'] = pij
    data['c'] = c
    data['SNR'] = SNR

    data['RHO'] = RHO
    data['Y'] = Y
    data['GAMMA'] = GAMMA
    data['PI'] = PI
    data['Z'] = Z
    data['X'] = X

    #Persistance
    data_file_url = path_data + '/' + fname + '.pickle'
    if verbose==True:
        print('Saving file to {:s} ... '.format(data_file_url))
    pickle.dump({'data': data}, open(data_file_url, 'wb'))
    if verbose == True:
        print('save succesful')

    return data

# ############### MAIN PROGRAM #####################
def main():
    if len(sys.argv) < 2:
         path_data = 'data'
         fname = 'demo_smlsbm'
    else:
        path_data = sys.argv[1]
        fname = sys.argv[2]

    data=  create_multistrata(
        Q= 4, N = 128, K = 30, M=3,
        pii = [0.6,0.4,0.125], c = 20,
        path_data = path_data,
        fname = fname,
        verbose = True)
    sys.exit()

if __name__ == '__main__':

    main()
