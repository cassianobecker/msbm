import numpy as np
import numpy.random as npr
import sys
import scipy.special as sp
import sklearn.metrics as skm
import pickle

################# CAVI UPDATE FUNCTIONS #########################

def update_Pi(mom, data, prior, par):

    strsum = 'km,kij,kmiq,kmjr->mqr'

    NEW_ALPHA = par['kappa']*(prior['ALPHA_0'] +
                       np.einsum(strsum, mom['MU'], data['X'],
                                 mom['TAU'], mom['TAU']) - 1.0) + 1.0

    NEW_BETA = par['kappa']*(prior['BETA_0'] +
                      np.einsum(strsum, mom['MU'], 1.0 - data['X'],
                                mom['TAU'], mom['TAU']) - 1.0) + 1.0

    return NEW_ALPHA, NEW_BETA


def update_Z(mom, data, prior, par):

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))

    S1 = np.einsum('km,mq,i->kmiq', mom['MU'], NUdiff, np.ones(par['N']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(par['N']), np.ones(par['N']), np.ones(par['K']))

    P1 = np.einsum('kij,mqr->mqrijk',
                   data['X'],
                   sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    S2 = np.einsum('km,kmjr,mqrijk->kmiq', mom['MU'], mom['TAU'], P1 + P2)

    NEW_TAU = (S1 + S2)

    NEW_TAU = np.exp(par['kappa']*(NEW_TAU - np.expand_dims(np.max(NEW_TAU, axis=3), axis=3)))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    return NEW_TAU


def update_Y(mom, data, prior, par):

    ZETAdiff = sp.psi(mom['ZETA']) - np.expand_dims(np.sum(sp.psi(mom['ZETA'])), axis=1)

    S1 = np.einsum('m,k->km', ZETAdiff, np.ones(par['K']))

    P1 = np.einsum('kij,mqr->mqrijk', data['X'], sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(par['N']), np.ones(par['N']), np.ones(par['K']))

    S2 = np.einsum('kmiq,kmjr,mqrijk->km', mom['TAU'], mom['TAU'], P1 + P2)

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))

    S3 = np.einsum('kmiq,mq->km', mom['TAU'], NUdiff)

    NEW_MU = (S1 + S2 + S3)

    NEW_MU = np.exp(par['kappa']*(NEW_MU - np.expand_dims(np.max(NEW_MU, axis=1), axis=1)))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)

    return NEW_MU


def update_gamma(mom, data, prior, par):

    NEW_NU = par['kappa']*(prior['NU_0'] +
                           np.einsum('km,kmiq->mq', mom['MU'], mom['TAU']) - 1.0) + 1.0

    return NEW_NU


def update_rho(mom, data, prior, par):

    NEW_ZETA = par['kappa']*(prior['ZETA_0'] +
                             np.einsum('km->m', mom['MU']) - 1.0) + 1.0

    return NEW_ZETA

################## AUXILIARY FUNCTIONS #################


def find_col(idc):

    col = [np.nonzero(idc[i, :])[0][0] for i in range(idc.shape[0])]

    return col


def adj_rand(tau, Z):

    rand_index = skm.adjusted_rand_score(find_col(Z), np.argmax(tau, axis=1))

    return rand_index


################# INITIALIZATION FUNCTIONS ###############

def load_data(data_file_url):

    print('\nLoading data from to {:s} ... '.format(data_file_url), end='')
    loaded = pickle.load(open(data_file_url, 'rb'))
    print('loaded.')

    return loaded['data'], loaded['par']


def init_moments(par):

    npr.seed(0)

    prior = dict()
    prior['ALPHA_0'] = 0.5
    prior['BETA_0'] = 0.5
    prior['NU_0'] = 0.5
    prior['ZETA_0'] = 0.5

    mom = dict()
    mom['ALPHA'] = npr.rand(par['M'], par['Q'], par['Q'])
    mom['BETA'] = npr.rand(par['M'], par['Q'], par['Q'])
    mom['NU'] = npr.rand(par['M'], par['Q'])
    mom['ZETA'] = npr.rand(par['M'])

    MU = npr.rand(par['K'], par['M'])
    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)
    mom['MU'] = MU

    TAU = npr.rand(par['K'], par['M'], par['N'], par['Q'])
    for k in range(par['K']):
        for m in range(par['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    mom['TAU'] = TAU

    return mom, prior

#################### MAIN CAVI PROGRAM #####################

def cavi_msbm(mom, data, prior, par):

    print('##############################################################')
    print('------------------- RUNNING CAVI FOR MSBM --------------------')
    print('                K = {:d}, M = {:d}, N = {:d}, Q = {:}'.
          format(par['K'], par['M'], par['N'], par['Q']))
    print('##############################################################')

    T = par['MAX_ITER']

    for t in range(T):

        par['kappa'] = float((t+10))/float((T+10))

        print('Iter: {:3d} of {:3d} (kappa = {:.4f}) ---  adj. rand index: {:+.3f}'
              .format(t+1, T, par['kappa'], adj_rand(mom['MU'], data['Y'])))

        ALPHA, BETA = update_Pi(mom, data, prior, par)
        mom['ALPHA'] = ALPHA
        mom['BETA'] = BETA

        TAU = update_Z(mom, data, prior, par)
        mom['TAU'] = TAU

        NU = update_gamma(mom, data, prior, par)
        mom['NU'] = NU

        MU = update_Y(mom, data, prior, par)
        mom['MU'] = MU

        ZETA = update_rho(mom, data, prior, par)
        mom['ZETA'] = ZETA

    print('\nFinished (maximum number of iterations).')

    return mom


def test_cavi(data_file_url, out_file_url):

    data, par = load_data(data_file_url)

    mom, prior = init_moments(par)

    par['MAX_ITER'] = 15

    results_mom = cavi_msbm(mom, data, prior, par)

    print('\nSaving results to {:s} ... '.format(out_file_url), end='')
    pickle.dump({'mom': results_mom}, open(out_file_url, 'wb'))
    print('Saved.')


def main():

    if len(sys.argv) < 3:

        path_data = 'data'
        fname = 'msbm1'
        data_file_url = path_data + '/' + fname + '.pickle'
        out_file_url = path_data + '/' + 'results_' + fname + '.pickle'

    else:
        data_file_url = sys.argv[1]
        out_file_url = sys.argv[2]

    test_cavi(data_file_url, out_file_url)


if __name__ == '__main__':
    main()