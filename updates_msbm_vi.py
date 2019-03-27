"""
Functions for updating the variational parameters for the variational
inference as well as computation of the ELBO
"""
import numpy as np
import scipy.special as sp
import pdb
# ################ UPDATE FUNCTIONS #########################


def update_Pi(mom, data, prior, par):

    str_sum = 'km, kij, kmiq, kmjr -> mqr'

    NEW_ALPHA = par['kappa']*(prior['ALPHA_0']
                              + np.einsum(str_sum, mom['MU'], data['X'],
                                          mom['TAU'], mom['TAU']) - 1.0) + 1.0

    NEW_BETA = par['kappa']*(prior['BETA_0']
                             + np.einsum(str_sum, mom['MU'], 1.0 - data['X'],
                                         mom['TAU'], mom['TAU']) - 1.0) + 1.0

    return NEW_ALPHA, NEW_BETA


def update_Z(mom, data, prior, par):
    Q = mom['ALPHA'].shape[1]
    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(Q)))

    S1 = np.einsum('km,mq,i->kmiq', mom['MU'], NU_diff, np.ones(data['N']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(data['N']), np.ones(data['N']), np.ones(data['K']))

    P1 = np.einsum('kij,mqr->mqrijk',
                   data['X'],
                   sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    S2 = np.einsum('km,kmjr,mqrijk->kmiq', mom['MU'], mom['TAU'], P1 + P2)

    NEW_LOG_TAU = par['kappa']*(S1 + S2)

    return NEW_LOG_TAU


def par_from_mom_TAU(mom, par):

    NEW_TAU = np.exp(mom['LOG_TAU'] - np.expand_dims(np.max(mom['LOG_TAU'], axis=3), axis=3))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    return NEW_TAU


def update_Y(mom, data, prior, par):
    Q = mom['ALPHA'].shape[1]
    ZETA_diff = sp.psi(mom['ZETA']) - sp.psi(sum(mom['ZETA']))

    S1 = np.einsum('m,k->km', ZETA_diff, np.ones(data['K']))

    P1 = np.einsum('kij,mqr->mqrijk', data['X'], sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(data['N']), np.ones(data['N']), np.ones(data['K']))

    S2 = np.einsum('kmiq,kmjr,mqrijk->km', mom['TAU'], mom['TAU'], P1 + P2)

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(Q)))

    S3 = np.einsum('kmiq,mq->km', mom['TAU'], NUdiff)

    NEW_LOG_MU = par['kappa']*(S1 + S2 + S3)

    return NEW_LOG_MU


def par_from_mom_MU(mom, par):

    NEW_MU = np.exp(mom['LOG_MU'] - np.expand_dims(np.max(mom['LOG_MU'], axis=1), axis=1))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)

    return NEW_MU


def update_gamma(mom, data, prior, par):

    NEW_NU = par['kappa']*(prior['NU_0'] + np.einsum('km,kmiq->mq', mom['MU'], mom['TAU']) - 1.0) + 1.0

    return NEW_NU


def update_rho(mom, data, prior, par):

    NEW_ZETA = par['kappa']*(prior['ZETA_0'] + np.einsum('km->m', mom['MU']) - 1.0) + 1.0

    return NEW_ZETA


# ################# COMPUTING THE ELBO #################

def elbo_x(mom, data, prior, par):

    # We use one line from update_z
    ALPHA_diff = sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])
    BETA_diff = sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA'])

    # We use the einsums from update_pi (and add ALPHA_diff, BETA_diff)
    str_sum = 'km,kmiq,kmjr,kij,mqr->'
    P1 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], data['X'], ALPHA_diff)
    P2 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], 1.0 - data['X'], BETA_diff)

    lb_x = P1 + P2

    return lb_x


def elbo_gamma(mom, data, prior, par):

    # We use NUdiff from update_z
    Q = mom['ALPHA'].shape[1]
    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(Q)))
    lb_gamma = np.einsum('mq->', (prior['NU_0'] - mom['NU'])*NU_diff)

    # Add the \Gamma terms (Not in the updates)
    gamma_sum_nu = sum(sp.gammaln(np.einsum('mq->m', mom['NU'])))
    sum_gamma_nu = np.einsum('mq->', sp.gammaln(mom['NU']))
    gamma_sum_nu0 = sum(sp.gammaln(np.einsum('mq->m', np.ones(mom['NU'].shape)*prior['NU_0'])))
    sum_gamma_nu0 = np.einsum('mq->', sp.gammaln(np.ones(mom['NU'].shape)*prior['NU_0']))

    lb_gamma = lb_gamma + gamma_sum_nu0 - sum_gamma_nu0 - gamma_sum_nu + sum_gamma_nu

    return lb_gamma


def elbo_rho(mom, data, prior, par):

    # We use ZETA_diff from update_y
    ZETA_diff = sp.psi(mom['ZETA']) - sp.psi(sum(mom['ZETA']))
    lb_rho = sum((prior['ZETA_0'] - mom['ZETA'])*ZETA_diff)

    # We add gamma terms (not in any update)
    gamma_sum_zeta = sp.gammaln(sum(mom['ZETA']))
    sum_gamma_zeta = sum(sp.gammaln(mom['ZETA']))
    gamma_sum_zeta0 = sp.gammaln(sum(prior['ZETA_0']*np.ones(mom['ZETA'].shape)))
    sum_gamma_zeta0 = sum(sp.gammaln(prior['ZETA_0']*np.ones(mom['ZETA'].shape)))

    lb_rho = lb_rho + gamma_sum_zeta0 - sum_gamma_zeta0 - gamma_sum_zeta + sum_gamma_zeta

    return lb_rho


def elbo_pi(mom, data, prior, par):

    # We use one line from update_Z
    lb_alpha = np.einsum('mqr->', (prior['ALPHA_0'] - mom['ALPHA'])
                         * (sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])))

    lb_beta = np.einsum('mqr->', (prior['BETA_0'] - mom['BETA'])
                        * (sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA'])))

    # We add the gamma terms
    gamma_sum_ab = np.einsum('mqr->', sp.gammaln(mom['ALPHA']+mom['BETA']))

    sum_gamma_ab = np.einsum('mqr->', sp.gammaln(mom['ALPHA']) + sp.gammaln(mom['BETA']))

    gamma_sum_ab0 = np.einsum('mqr->', sp.gammaln(prior['ALPHA_0']*np.ones(mom['ALPHA'].shape)
                                                  + prior['BETA_0']*np.ones(mom['BETA'].shape)))

    sum_gamma_ab0 = np.einsum('mqr->', sp.gammaln(prior['ALPHA_0']*np.ones(mom['ALPHA'].shape))
                              + sp.gammaln(prior['BETA_0']*np.ones(mom['BETA'].shape)))

    lb_pi = lb_alpha + lb_beta + gamma_sum_ab0 - sum_gamma_ab0 - gamma_sum_ab + sum_gamma_ab

    return lb_pi


def elbo_y(mom, data, prior, par):

    # We use ZETA_diff from update_y
    ZETA_diff = sp.psi(mom['ZETA']) - sp.psi(sum(mom['ZETA']))

    lb_y = np.einsum('km,m->', mom['MU'], ZETA_diff) - np.einsum('km->', sp.xlogy(mom['MU'], mom['MU']))

    return lb_y


def elbo_z(mom, data, prior, par):
    Q = mom['ALPHA'].shape[1]
    # We use NU_diff from update_z
    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(Q)))

    P1 = np.einsum('km,kmiq,mq->', mom['MU'], mom['TAU'], NU_diff)

    P2 = np.einsum('km,kmiq->', mom['MU'], sp.xlogy(mom['TAU'], mom['TAU']))

    lb_z = P1 - P2

    return lb_z


def compute_elbo(mom, data, prior, par):

    elbo = dict()
    elbo['x'] = elbo_x(mom, data, prior, par)
    elbo['rho'] = elbo_rho(mom, data, prior, par)
    elbo['pi'] = elbo_pi(mom, data, prior, par)
    elbo['gamma'] = elbo_gamma(mom, data, prior, par)
    elbo['y'] = elbo_y(mom, data, prior, par)
    elbo['z'] = elbo_z(mom, data, prior, par)

    return sum(elbo.values())


def compute_elbos(mom, data, prior, par, elbos = None):
    if elbos is None:
        elbos = dict()
    if len(list(elbos.keys())) == 0:
        elbos['x'] = list()
        elbos['rho'] = list()
        elbos['pi'] = list()
        elbos['gamma'] = list()
        elbos['y'] = list()
        elbos['z'] = list()
        elbos['all'] = list()

    elbos['x'].append(elbo_x(mom, data, prior, par))
    elbos['rho'].append(elbo_rho(mom, data, prior, par))
    elbos['pi'].append(elbo_pi(mom, data, prior, par))
    elbos['gamma'].append(elbo_gamma(mom, data, prior, par))
    elbos['y'].append(elbo_y(mom, data, prior, par))
    elbos['z'].append(elbo_z(mom, data, prior, par))
    
    elbo = sum([elbos[key][-1] for key in elbos.keys() if key not in ['all']])
    elbos['all'].append(elbo)

    return elbos
