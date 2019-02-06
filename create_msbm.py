import numpy as np
import numpy.random as npr
import sys
import pickle

def accept_prob(e,eprime,T):
    """
    accept_prob acceptance probability for every step of
    the simulated annealing. We accept the new state whenever
    it is a lower energy state, otherwise we accept with a probability
    that decays exponentially in proportion to the difference of energies
    and is affected by a temperature parameter. 
    Parameters
    ----------
    e : double 
        energy of current state
    eprime : double
        energy of proposed state
    T : double
    	current temperature parameter
    """
    if eprime < e:
        p = 1
    else:
        p = exp(-(eprime - e)/T)
    return p


# ############# SAMPLING FUNCTIONS #######################
# We sample gamma and pi from Dirichlet and Beta distributions
# respectively by specifying means and variances. Optionally, we
# can constraint on a sample Dt f-divergence, which will determine
# asymptotic detectability. For the constrained sampling we use simulated annealing.


def get_gamma_pi(
	Q = 4, sampling = 'unconstrained',
	vdir, mii, vii, mij, vij,
	maxIter = 100, tol = 1e-07
):
    """
    get_gamma_pi a function that samples gamma (the community
    weights or community importance) from a homogeneous Dirichlet
	distribution (which has mean, a uniform distribution)
	and pi (the connection probabilities), by specifying
	means and variances. Optionally, we can constraint the sample to 
	have a target Dt f-divergence (E. Abbe and C. Sandon. 2015)
	which will determine asymptotic detectability. For the constrained
	sampling we use simulated annealing.
    Parameters
    ----------
    Q : int 
        the number of communities. Defaults to 4
    sampling : str
        the type of sampling to be made. The options are
        'deterministic', 'unconstrained' and 'constrained'
        which correspond to deterministic annealing 
    vdir : double
    	variance of the homogeneous dirichlet distribution
    mii : double
    	mean of the whithin community beta distributions
    vii : double
    	variance of the whithin community beta distributions
    mij : double
    	mean of the across community beta distributions
    vii : double
    	variance of the across community beta distributions
    maxIter : int 
    	maximum number of iterations for the simulated annealing
    """
    #Obtain the alphas with target var and assuming a uniform mean
    if sampling not in {'constrained','unconstrained'}:
    	sys.exit("Unsupported sampling type")

    alpha = ((Q-1)/(vdir*Q^2)+1)/Q
    gamma = npr.dirichlet(np.repeat(alpha, Q),1)
    
    #Beta parameters alpha_ii and beta_ii
    alpha_ii = (mii^2*(1-mii) - mii*vii)/vii
    beta_ii  = (1/vii)*(mii*(1-mii) - vii)*(1-mii)
    #Beta parameters alpha_ij and beta_ij
    alpha_ij = (mij^2*(1-mij) - mij*vij)/vij
    beta_ij  = (1/vij)*(mij*(1-mij) - vij)*(1-mij)

    pi = np.zeros((Q,Q))
    for i in range(Q):
    	for j in range(Q):
    		if i == j:
    			pi[i,j] = npr.beta(alpha_ii,beta_ii)/2

    		if i < j:
    			pi[i,j] = npr.beta(alpha_ij,beta_ij)

    pi = pi + np.transpose(pi)

    if sampling == 'unconstrained':
    	return gamma, pi

    print('Beggining constrained sampling with:{:d} iterations'.format(numIter))
    print('---------------------------------------------------')
	print('')    

def get_pi(m):

    # M = 0.3
    # H = 0.7
    # L = 0.1

    Pi = list()

#    Pi.append(np.array([[H, L, L],
#                        [L, H, L],
#                        [L, L, H]]))

    Pi.append(np.array([[0.2176, 0.0185, 0.0836, 0.1015],
                        [0.0614, 0.4529, 0.0327, 0.0008],
                        [0.0048, 0.0642, 0.4515, 0.1021],
                        [0.0192, 0.0403, 0.0800, 0.4061]]))

#    Pi.append(np.array([[H, M, M],
#                        [M, H, M],
#                        [M, M, H]]))

    Pi.append(np.array([[0.6789, 0.0293, 0.0437, 0.1759],
                        [0.1498, 0.6418, 0.0317, 0.0043],
                        [0.0057, 0.1007, 0.4528, 0.1595],
                        [0.2032, 0.1081, 0.3031, 0.4716]]))

#    Pi.append(np.array([[M, L, L],
#                        [L, M, L],
#                        [L, L, M]]))

    Pi.append(np.array([[0.5706, 0.0187, 0.0697, 0.0196],
                        [0.0108, 0.3748, 0.0352, 0.0887],
                        [0.0047, 0.0349, 0.2650, 0.1951],
                        [0.0008, 0.0432, 0.0012, 0.4281]]))

#    Pi.append(np.array([[M, M, M],
#                        [M, M, M],
#                        [M, M, M]]))

    Pi.append(np.array([[0.3659, 0.1162, 0.0579, 0.0148],
                        [0.1144, 0.4836, 0.0219, 0.0475],
                        [0.0627, 0.0016, 0.4275, 0.1200],
                        [0.0803, 0.1355, 0.0054, 0.2231]]))

    return Pi[m]


def get_rho():

    rho = np.array([0.2, 0.5, 0.7, 0.3])

    return rho/np.sum(rho)


def sample_Y(rho, K):

    Y = npr.multinomial(1, rho, K)

    return Y.astype(float)


def sample_Z(gamma, N):

    Z = npr.multinomial(1, gamma, N)

    return Z.astype(float)

def sample_X_und(Pi, Z):

    N = Z.shape[0]
    X = np.zeros((N, N))
    for i in range(N):
            zi = find_row(Z[i, :])
            for j in range(i, N):
                zj = find_row(Z[j, :])
                X[i, j] = npr.binomial(1, Pi[zi, zj])
                X[j, i] = X[i, j]

    return X.astype(float)


# ############### MODEL CREATION ################


def create_msbm(Q, N, M, K):

    print('---- Creating MSBM mode with N = {:d} and K = {:d} ------'
          .format(N, K))

    RHO = get_rho()

    Y = sample_Y(RHO, K)

    GAMMA = np.zeros((M, Q))
    for m in range(M):
        GAMMA[m, :] = get_gamma(m)

    PI = np.zeros((M, Q, Q))
    for m in range(M):
        PI[m, :] = get_pi(m)

    Z = np.zeros((K, N, Q))
    X = np.zeros((K, N, N))
    for k in range(K):
        m = find_row(Y[k, :])
        Z[k, :] = sample_Z(GAMMA[m, :], N)
        X[k, :] = sample_X_und(PI[m, :], Z[k, :])

    par = dict()
    par['Q'] = Q
    par['N'] = N
    par['M'] = M
    par['K'] = K

    data = dict()
    data['RHO'] = RHO
    data['Y'] = Y
    data['GAMMA'] = GAMMA
    data['PI'] = PI
    data['Z'] = Z
    data['X'] = X

    return data, par


def find_row(x):

    return np.nonzero(x)[0][0]

# ############### MAIN PROGRAM #####################


def main():

    # initialize file names
    if len(sys.argv) < 2:
        path_data = 'data'
        fname = 'msbm1'
        data_file_url = path_data + '/' + fname + '.pickle'

    else:
        data_file_url = sys.argv[1]

    # number of classes
    Q = 4
    # number of nodes
    N = 150
    # number of models
    M = 4
    # number of networks
    K = 80

    # sample model
    data, par = create_msbm(Q, N, M, K)

    # save to file
    print('Saving file to {:s} ... '.format(data_file_url), end='')
    pickle.dump({'data': data, 'par': par}, open(data_file_url, 'wb'))
    print('saved.')


if __name__ == '__main__':

    main()
