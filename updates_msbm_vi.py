"""
Functions for updating the variational parameters for the variational
inference as well as computation of the ELBO
"""
import numpy as np
import scipy.special as sp
import pdb
from scipy.stats import beta
from scipy.stats import dirichlet
# ################ UPDATE FUNCTIONS #########################


def update_Pi(data, prior, hyper, mom, par, remove_self_loops= True):

    str_sum = 'km, kij, kmiq, kmjr -> mqr'
    NEW_ALPHA = par['kappa']*(prior['ALPHA_0']
                              + np.einsum(str_sum, mom['MU'], data['X'],
                                          mom['TAU'], mom['TAU']) - 1.0) + 1.0
    if remove_self_loops:
        NON_EDGES = data['NON_X']
    else:
        NON_EDGES = 1.0 - data['X']

    NEW_BETA = par['kappa']*(prior['BETA_0']
                             + np.einsum(str_sum, mom['MU'], NON_EDGES,
                                         mom['TAU'], mom['TAU']) - 1.0) + 1.0

    return NEW_ALPHA, NEW_BETA

def Pi_from_mom(mom):

    Pi_estimate = beta.stats(mom['ALPHA'],mom['BETA'],moments='m')
    return Pi_estimate

def update_Z(data, prior, hyper, mom, par, remove_self_loops= True):

    if remove_self_loops:
        NON_EDGES = data['NON_X']
    else:
        NON_EDGES = 1.0 - data['X']

    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(hyper['Q'])))

    S1 = np.einsum('km,mq,i->kmiq', mom['MU'], NU_diff, np.ones(data['N']))

    P_EDGES = np.einsum('kij,mqr->mqrijk', data['X'], sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA']))

    P_NONEDGES = np.einsum('kij,mqr->mqrijk', NON_EDGES, sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']))

    S2 = np.einsum('km,kmjr,mqrijk->kmiq', mom['MU'], mom['TAU'], P_EDGES + P_NONEDGES)

    LOG_TAU = (S1 + S2)

    NEW_TAU = np.exp(LOG_TAU - np.expand_dims(np.max(LOG_TAU, axis=1), axis=1))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=1), axis=1)

    NEW_LOG_TAU = par['kappa']*(np.log(NEW_TAU))

    return NEW_LOG_TAU


def TAU_from_LOG_TAU(mom, par):

    NEW_TAU = np.exp(mom['LOG_TAU'] - np.expand_dims(np.max(mom['LOG_TAU'], axis=3), axis=3))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    return NEW_TAU

def update_Y(data, prior, hyper, mom, par, remove_self_loops= True):

    if remove_self_loops:
        NON_EDGES = data['NON_X']
    else:
        NON_EDGES = 1.0 - data['X']

    ZETA_diff = sp.psi(mom['ZETA']) - sp.psi(sum(mom['ZETA']))

    S1 = np.einsum('m,k->km', ZETA_diff, np.ones(data['K']))

    P_EDGES = np.einsum('kij,mqr->mqrijk', data['X'], sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA']))

    P_NONEDGES = np.einsum('kij,mqr->mqrijk', NON_EDGES, sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']))

    S2 = np.einsum('kmiq,kmjr,mqrijk->km', mom['TAU'], mom['TAU'], P_EDGES + P_NONEDGES)

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(hyper['Q'])))

    S3 = np.einsum('kmiq,mq->km', mom['TAU'], NUdiff)
    #We obtain the unnormalized natural parameters.
    LOG_MU = (S1 + S2 + S3)

    NEW_MU = np.exp(LOG_MU - np.expand_dims(np.max(LOG_MU, axis=1), axis=1))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)
    #From this we can recover the normalized natural parameters
    NEW_LOG_MU = par['kappa']*(np.log(NEW_MU))

    return NEW_LOG_MU


def par_from_mom_MU(mom, par): #This works for CAVI, but we need another method for natgrad

    NEW_MU = np.exp(mom['LOG_MU'] - np.expand_dims(np.max(mom['LOG_MU'], axis=1), axis=1))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)

    return NEW_MU

def update_gamma(data, prior, hyper, mom, par, remove_self_loops=True):

    NEW_NU = par['kappa']*(prior['NU_0'] + np.einsum('km,kmiq->mq', mom['MU'], mom['TAU']) - 1.0) + 1.0

    return NEW_NU

def Gamma_from_mom(mom):

    Gamma_estimate = [dirichlet.mean(mom['NU'][m,:]) for m in range(mom['NU'].shape[0]) ]
    return Gamma_estimate

def update_rho(data, prior, hyper, mom, par, remove_self_loops=True):

    NEW_ZETA = par['kappa']*(prior['ZETA_0'] + np.einsum('km->m', mom['MU']) - 1.0) + 1.0

    return NEW_ZETA


# ################# COMPUTING THE ELBO #################

def elbo_x(data, prior, hyper, mom, par):

    # We use one line from update_z
    ALPHA_diff = sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])
    BETA_diff = sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA'])

    # We use the einsums from update_pi (and add ALPHA_diff, BETA_diff)
    str_sum = 'km,kmiq,kmjr,kij,mqr->'
    P1 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], data['X'], ALPHA_diff)
    P2 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], 1.0 - data['X'], BETA_diff)

    lb_x = P1 + P2

    return lb_x


def elbo_gamma(data, prior, hyper, mom, par):

    # We use NUdiff from update_z

    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(hyper['Q'])))
    lb_gamma = np.einsum('mq->', (prior['NU_0'] - mom['NU'])*NU_diff)

    # Add the \Gamma terms (Not in the updates)
    gamma_sum_nu = sum(sp.gammaln(np.einsum('mq->m', mom['NU'])))
    sum_gamma_nu = np.einsum('mq->', sp.gammaln(mom['NU']))
    gamma_sum_nu0 = sum(sp.gammaln(np.einsum('mq->m', np.ones(mom['NU'].shape)*prior['NU_0'])))
    sum_gamma_nu0 = np.einsum('mq->', sp.gammaln(np.ones(mom['NU'].shape)*prior['NU_0']))

    lb_gamma = lb_gamma + gamma_sum_nu0 - sum_gamma_nu0 - gamma_sum_nu + sum_gamma_nu

    return lb_gamma


def elbo_rho(data, prior, hyper, mom, par):

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


def elbo_pi(data, prior, hyper, mom, par):

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


def elbo_y(data, prior, hyper, mom, par):

    # We use ZETA_diff from update_y
    ZETA_diff = sp.psi(mom['ZETA']) - sp.psi(sum(mom['ZETA']))

    lb_y = np.einsum('km,m->', mom['MU'], ZETA_diff) - np.einsum('km->', sp.xlogy(mom['MU'], mom['MU']))

    return lb_y


def elbo_z(data, prior, hyper, mom, par):

    # We use NU_diff from update_z
    NU_diff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(hyper['Q'])))

    P1 = np.einsum('km,kmiq,mq->', mom['MU'], mom['TAU'], NU_diff)

    P2 = np.einsum('km,kmiq->', mom['MU'], sp.xlogy(mom['TAU'], mom['TAU']))

    lb_z = P1 - P2

    return lb_z


def compute_elbo(data, prior, hyper, mom, par):

    elbo = dict()
    elbo['x'] = elbo_x(data, prior, hyper, mom, par)
    elbo['rho'] = elbo_rho(data, prior, hyper, mom, par)
    elbo['pi'] = elbo_pi(data, prior, hyper, mom, par)
    elbo['gamma'] = elbo_gamma(data, prior, hyper, mom, par)
    elbo['y'] = elbo_y(data, prior, hyper, mom, par)
    elbo['z'] = elbo_z(data, prior, hyper, mom, par)

    return sum(elbo.values())


def compute_elbos(data, prior, hyper, mom, par, elbos = None):
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

    elbos['x'].append(elbo_x(data, prior, hyper, mom, par))
    elbos['rho'].append(elbo_rho(data, prior, hyper, mom, par))
    elbos['pi'].append(elbo_pi(data, prior, hyper, mom, par))
    elbos['gamma'].append(elbo_gamma(data, prior, hyper, mom, par))
    elbos['y'].append(elbo_y(data, prior, hyper, mom, par))
    elbos['z'].append(elbo_z(data, prior, hyper, mom, par))
    
    elbo = sum([elbos[key][-1] for key in elbos.keys() if key not in ['all']])
    elbos['all'].append(elbo)

    return elbos
