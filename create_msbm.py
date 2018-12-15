import numpy as np
import numpy.random as npr
import sys
import pickle

# ############# SAMPLING FUNCTIONS #######################


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


def get_gamma(m):

    gamma = list()

    gamma.append(np.array([0.4773, 0.3043, 0.1458, 0.0725]))
    gamma.append(np.array([0.2650, 0.2580, 0.2499, 0.2271]))  # Tweak to make it easier
    gamma.append(np.array([0.3597, 0.2641, 0.2617, 0.1145]))
    gamma.append(np.array([0.3540, 0.3082, 0.2169, 0.1209]))

    return gamma[m]/np.sum(gamma[m])


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
