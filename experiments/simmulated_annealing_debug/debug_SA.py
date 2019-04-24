import os, sys
import pickle
import numpy as np
import scipy as sp
import numpy.random as npr
import pdb
import matplotlib.pyplot as plt
from scipy.stats import beta

sys.path.insert(0, '../..')
import generate_msbm as gn

def main():
    Q = 4
    vdir = 0.015
    c = 16
    SNR = 1.05
    n = 128
    alpha = ((Q - 1) / (vdir * Q ** 2) + 1) / Q
    gamma = npr.dirichlet(np.repeat(alpha, Q), 1)
    #gamma = np.repeat(1/Q, Q).reshape((1,4))
    alpha_ii, beta_ii, alpha_ij, beta_ij = gn.get_beta_moms(c, n, Q, SNR + 0.55, scale = 0.5, debugging = True)
    #Print means of betas
    print("Actual mii: {}, Actual mij: {}".format( \
        beta.mean(alpha_ii, beta_ii),\
        beta.mean(alpha_ij, beta_ij) ))
    
    #Plot the densities:
    x = np.linspace(beta.ppf(0.01, alpha_ij, beta_ij), beta.ppf(0.99, alpha_ij, beta_ij), 200)
    rv = beta(alpha_ij, beta_ij)
    plt.plot(x, rv.pdf(x), 'k-', lw=2, label='diag density')
    #plt.pause(5)
    plt.clf()
    plt.close()

    x = np.linspace(beta.ppf(0.01, alpha_ii, beta_ii), beta.ppf(0.99, alpha_ii, beta_ii), 200)
    rv = beta(alpha_ii, beta_ii)
    plt.plot(x, rv.pdf(x), 'k-', lw=2, label='diag density')
    #plt.pause(5)
    #Compare random prototype and deterministic prototype:
    ALPHA_0 = np.ones((Q, Q))*alpha_ij + np.diag( np.repeat(alpha_ii - alpha_ij,Q))
    BETA_0 = np.ones((Q,Q))*beta_ij + np.diag( np.repeat(beta_ii - beta_ij,Q))

    pi = npr.beta(ALPHA_0,BETA_0)
    #Symmetrize
    pi = np.tril(pi) + np.transpose(np.tril(pi,-1))

    pi_det = beta.mean(ALPHA_0, BETA_0)
    print(pi)
    print("\n")
    print(pi_det)

    #Now we obtain the CH-Divergence of each:
    cSNR, com_i, com_j = gn.getSNR(gamma, pi* (n / np.log(n)))
    print("random Pi has CH_Div: {}, attained by columns {} and {}".format(cSNR, com_i, com_j))
    print("\n")

    cSNR, com_i, com_j = gn.getSNR(gamma, pi_det* (n / np.log(n)))
    print("deterministic Pi has CH_Div: {}, attained by columns {} and {}".format(cSNR, com_i, com_j))
    sys.exit()


if __name__ == '__main__':
    main()
