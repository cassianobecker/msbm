import numpy as np
import numpy.random as npr

#TO DO (fix all of this so that it takes data)
def init_moments(data, hyper):
    mom = dict()

    mode = 'random'

    mom['ALPHA'] = init_ALPHA(hyper, mode)
    mom['BETA'] = init_BETA(hyper, mode)
    mom['NU'] = init_NU(hyper, mode)
    mom['ZETA'] = init_ZETA(hyper, mode)
    mom['MU'] = init_MU(hyper, mode)
    mom['LOG_MU'] = np.log(mom['MU'])
    mom['TAU'] = init_TAU(hyper, mode)
    mom['LOG_TAU'] = np.log(mom['TAU'])

    return mom


def init_NU(hyper, mode='random'):

    if mode == 'random':
        NU = npr.rand(hyper['M'], hyper['Q'])
    if mode == 'uniform':
        NU = np.ones((hyper['M'], hyper['Q'])) / hyper['Q']

    return NU


def init_ZETA(hyper, mode='random'):

    if mode == 'random':
        ZETA = npr.rand(hyper['M'])
    if mode == 'uniform':
        ZETA = np.ones(hyper['M'])/hyper['M']

    return ZETA


def init_ALPHA(hyper, mode='random'):

    if mode == 'random':
        ALPHA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if mode == 'uniform':
        ALPHA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return ALPHA


def init_BETA(hyper, mode='random'):

    if mode == 'random':
        BETA = npr.rand(hyper['M'], hyper['Q'], hyper['Q'])
    if mode == 'uniform':
        BETA = 0.5 * np.ones((hyper['M'], hyper['Q'], hyper['Q']))

    return BETA


def init_MU(hyper, mode='random'):

    if mode == 'random':
        MU = npr.rand(hyper['K'], hyper['M'])
    if mode == 'uniform':
        MU = np.ones((hyper['K'], hyper['M'])) / hyper['M']

    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)

    return MU


def init_TAU(hyper, mode='random'):

    if mode == 'random':
        TAU = npr.rand(hyper['K'], hyper['M'], hyper['N'], hyper['Q'])
    if mode == 'uniform':
        TAU = np.ones((hyper['K'], hyper['M'], hyper['N'], hyper['Q']))

    for k in range(hyper['K']):
        for m in range(hyper['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    return TAU
