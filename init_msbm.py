import numpy as np
import numpy.random as npr


def init_moments(par):

    #npr.seed(1)

    prior = dict()
    prior['ALPHA_0'] = 0.5
    prior['BETA_0'] = 0.5
    prior['NU_0'] = 0.5
    prior['ZETA_0'] = 0.5

    mom = dict()

    mode = 'random'

    mom['ALPHA'] = init_ALPHA(par, mode)
    mom['BETA'] = init_BETA(par, mode)
    mom['NU'] = init_NU(par, mode)
    mom['ZETA'] = init_ZETA(par, mode)
    mom['MU'] = init_MU(par, mode)
    mom['LOG_MU'] = np.log(mom['MU'])
    mom['TAU'] = init_TAU(par, mode)
    mom['LOG_TAU'] = np.log(mom['TAU'])

    return mom, prior


def init_NU(par, mode='random'):

    if mode == 'random':
        NU = npr.rand(par['M'], par['Q'])
    if mode == 'uniform':
        NU = np.ones((par['M'], par['Q'])) / par['Q']

    return NU


def init_ZETA(par, mode='random'):

    if mode == 'random':
        ZETA = npr.rand(par['M'])
    if mode == 'uniform':
        ZETA = np.ones(par['M'])/par['M']


    return ZETA


def init_ALPHA(par, mode='random'):

    if mode == 'random':
        ALPHA = npr.rand(par['M'], par['Q'], par['Q'])
    if mode == 'uniform':
        ALPHA = 0.5 * np.ones((par['M'], par['Q'], par['Q']))

    return ALPHA


def init_BETA(par, mode='random'):

    if mode == 'random':
        BETA = npr.rand(par['M'], par['Q'], par['Q'])
    if mode == 'uniform':
        BETA = 0.5 * np.ones((par['M'], par['Q'], par['Q']))

    return BETA


def init_MU(par, mode='random'):

    if mode == 'random':
        MU = npr.rand(par['K'], par['M'])
    if mode == 'uniform':
        MU = np.ones((par['K'], par['M'])) / par['M']

    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)

    return MU


def init_TAU(par, mode='random'):

    if mode == 'random':
        TAU = npr.rand(par['K'], par['M'], par['N'], par['Q'])
    if mode == 'uniform':
        TAU = np.ones((par['K'], par['M'], par['N'], par['Q']))

    for k in range(par['K']):
        for m in range(par['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    return TAU

def init_moments_truth(par, data):

    npr.seed(1)

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

    eps = 1.e-6

    mom['MU'] = data['Y']
    mom['LOG_MU'] = np.log(mom['MU'] + eps)

    mom['TAU'] = np.einsum('ijk,l->iljk', data['Z'], np.ones(par['M']))
    mom['LOG_TAU'] = np.log(mom['TAU'] + eps)

    return mom, prior
