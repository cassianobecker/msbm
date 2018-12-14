import numpy as np
import numpy.random as npr


def init_moments(par):

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

    MU = npr.rand(par['K'], par['M'])
    MU = MU / np.expand_dims(np.sum(MU, axis=1), axis=1)
    mom['MU'] = MU
    mom['LOG_MU'] = np.log(MU)

    TAU = npr.rand(par['K'], par['M'], par['N'], par['Q'])
    for k in range(par['K']):
        for m in range(par['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    mom['TAU'] = TAU
    mom['LOG_TAU'] = np.log(TAU)

    return mom, prior


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
