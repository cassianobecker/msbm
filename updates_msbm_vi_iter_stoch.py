"""
Functions for updating the variational parameters for the variational inference
"""
import numpy as np
import numpy.random as npr
import scipy.special as sp
import pdb
from scipy.stats import beta
from scipy.stats import dirichlet
# ################ UPDATE FUNCTIONS #########################


def update_Pi(data, prior, hyper, mom, par):

    NEW_ALPHA = np.zeros((data['M'], data['Q'], data['Q']))
    NEW_BETA = np.zeros((data['M'], data['Q'], data['Q']))

    for m in range(data['M']):
        for q in range(data['Q']):
            for r in range(data['Q']):

                [aqr, bqr] = get_sum_a_b_pi(m, q, r, data, mom, par, prior)

                NEW_ALPHA[m, q, r] = prior['ALPHA_0'] + aqr

                NEW_BETA[m, q, r] = prior['BETA_0'] + bqr

    return NEW_ALPHA, NEW_BETA

def Pi_from_mom(mom):

    Pi_estimate = beta.stats(mom['ALPHA'],mom['BETA'],moments='m')
    return Pi_estimate

def get_sum_a_b_pi(m, q, r, data, mom, par, prior):

    aqr = 0
    bqr = 0
    b = int(150)
    c = int(15)
    m_size = len(data['strata'][0])
    #sample a subset of the networks K
    nets = npr.choice(data['K'], np.max(c, data['K']), replace = False)
    #For some nodes, flip a coin to decide if you sample a link set or non-link set

    for k in nets:
        nodes = npr.choice(data['N'], b, replace = False)
        for i in nodes:
                coin = npr.binomial(1, .5, 1)
                if coin == 1:
                    for j in data['strata'][k*data['N'] + i][0]:
                        aqr = aqr + ((data['K']/c)*data['N']/b)*data['X'][k, i, j]*mom['MU'][k, m] \
                              * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]

                        bqr = bqr + ((data['K']/c)*data['N']/b)*(1 - data['X'][k, i, j])*mom['MU'][k, m] \
                            * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]
                else:
                    m_ = npr.choice(range(1, m_size), 1)[0]
                    for j in data['strata'][k*data['N'] + i][m_]:
                        aqr = aqr + ((data['K']/c)*(m_size-1)*data['N']/b)*data['X'][k, i, j]*mom['MU'][k, m] \
                            * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]

                        bqr = bqr + ((data['K']/c)*(m_size-1)*data['N']/b)*(1 - data['X'][k, i, j])*mom['MU'][k, m] \
                            * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]
    return aqr, bqr


# ###########################################################

def update_Z(data, prior, hyper, mom, par, remove_self_loops = True):

    LOG_TAU = mom['LOG_TAU'].copy()
    b = int(150)
    m_size = len(data['strata'][0])
    nets = npr.choice(data['K'], min(int(b/4), data['K']), replace = False)
    for m in range(data['M']):
        for k in nets:
            nodes = npr.choice(data['N'], b, replace = False)
            for i in nodes:
                coin = npr.binomial(1, .5, 1)
                if coin == 1:
                    for q in range(data['Q']):
                        LOG_TAU[k, m, i, q] = get_log_tau_kmiq(
                            i, q,
                            data['N'], data['Q'], data['X'][k, :, :][:, data['strata'][k*data['N'] + i][0]],
                            mom['TAU'][k, m, :, :][data['strata'][k*data['N'] + i][0], :],
                            mom['ALPHA'][m, :, :],
                            mom['BETA'][m, :, :],
                            mom['NU'][m, :],
                            mom['MU'][k, m],
                            2)
                else:
                    #choose a batch of non-edges
                    m_ = npr.choice(range(1, m_size), 1)[0]
                    for q in range(data['Q']):
                        LOG_TAU[k, m, i, q] = get_log_tau_kmiq(
                            i, q,
                            data['N'], data['Q'], data['X'][k, :, :][:, data['strata'][k*data['N'] + i][m_]],
                            mom['TAU'][k, m, :, :][data['strata'][k*data['N'] + i][m_], :],
                            mom['ALPHA'][m, :, :],
                            mom['BETA'][m, :, :],
                            mom['NU'][m, :],
                            mom['MU'][k, m],
                            2*(m_size-1))   

   
    NEW_TAU = np.exp(LOG_TAU - np.expand_dims(np.max(LOG_TAU, axis=3), axis=3))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    NEW_LOG_TAU = np.log(NEW_TAU)                

    return NEW_LOG_TAU


def get_log_tau_kmiq(i, q, N, Q, Xk, tau_km, a_m, b_m, nu_m, mu_km, rescale):

    pnuq = sp.psi(nu_m[q])
    psnuq = sp.psi(np.sum(nu_m))

    log_tau_km_x = np.zeros((N, Q))
    log_tau_km_non_x = np.zeros((N, Q))

    for j in range(Xk.shape[1]):
        for r in range(Q):

            if not j==i:

                x_kij = Xk[i, j]
                tau_kmjr = tau_km[j, r] #FOUND THE BUG

                psi_a_mqr = sp.psi(a_m[q, r])
                psi_b_mqr = sp.psi(b_m[q, r])
                psi_ab_mqr = sp.psi(a_m[q, r] + b_m[q, r])

                log_tau_km_x[j, r] = tau_kmjr * (x_kij * (psi_a_mqr - psi_ab_mqr))
                log_tau_km_non_x[j, r] = tau_kmjr * ((1-x_kij)*(psi_b_mqr - psi_ab_mqr))

    log_tau_kmiq = (pnuq - psnuq) + rescale * mu_km*(np.sum(np.sum(log_tau_km_x)) + np.sum(np.sum(log_tau_km_non_x)))

    return log_tau_kmiq

def TAU_from_LOG_TAU(mom, par):

    NEW_TAU = np.exp(mom['LOG_TAU'] - np.expand_dims(np.max(mom['LOG_TAU'], axis=3), axis=3))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    return NEW_TAU
# ###################################################################################

def update_Y(data, prior, hyper, mom, par, remove_self_loops = True):

    LOG_MU = np.zeros((data['K'], data['M']))

    for k in range(data['K']):
        for m in range(data['M']):

            LOG_MU[k, m] = get_log_mu_km(m,
                                             data['N'], data['Q'], data['X'][k, :, :],
                                             mom['TAU'][k, m, :, :],
                                             mom['ALPHA'][m, :, :],
                                             mom['BETA'][m, :, :],
                                             mom['ZETA'],
                                             mom['NU'][m, :],
                                             remove_self_loops)

    NEW_MU = np.exp(LOG_MU - np.expand_dims(np.max(LOG_MU, axis=1), axis=1))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)
    #From this we can recover the normalized natural parameters
    NEW_LOG_MU = np.log(NEW_MU)

    return NEW_LOG_MU

def par_from_mom_MU(mom, par):

    NEW_MU = np.exp(mom['LOG_MU'] - np.expand_dims(np.max(mom['LOG_MU'], axis=1), axis=1))

    NEW_MU2 = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)

    return NEW_MU2

def get_log_mu_km(m, N, Q, Xk, tau_km, a_m, b_m, zeta, nu_m, remove_self_loops = True):

    psi_zeta_m = sp.psi(zeta[m])
    psi_sum_zeta = sp.psi(np.sum(zeta))
    log_mu_km = psi_zeta_m - psi_sum_zeta

    for i in range(N):
        for j in range(N):
            if not (remove_self_loops and j == i):
                for q in range(Q):
                    for r in range(Q):
                        x_kij = Xk[i, j]

                        tau_kmiq = tau_km[i, q]
                        tau_kmjr = tau_km[j, r]

                        psi_a_mqr = sp.psi(a_m[q, r])
                        psi_b_mqr = sp.psi(b_m[q, r])
                        psi_ab_mqr = sp.psi(a_m[q, r] + b_m[q, r])

                        log_mu_km = log_mu_km + tau_kmiq * tau_kmjr * (
                                    x_kij * (psi_a_mqr - psi_b_mqr) + psi_b_mqr - psi_ab_mqr)

    return log_mu_km


# ########################################################################

def update_gamma(data, prior, hyper, mom, par):

    NEW_NU = np.zeros((data['M'], data['Q']))

    for m in range(data['M']):
        for q in range(data['Q']):
            nu_mq = 0

            for k in range(data['K']):
                nu_mq = nu_mq + mom['MU'][k, m] * np.sum(mom['TAU'][k, m, :, q])

            NEW_NU[m, q] = prior['NU_0'] + nu_mq

    return NEW_NU

def Gamma_from_mom(mom):

    Gamma_estimate = [dirichlet.mean(mom['NU'][m,:]) for m in range(mom['NU'].shape[0]) ]
    return Gamma_estimate
# ########################################################################

def update_rho(data, prior, hyper, mom, par):

    NEW_ZETA = np.zeros((data['M']))

    for m in range(data['M']):
        NEW_ZETA[m] = prior['ZETA_0'] + np.sum(mom['MU'][:, m])

    return NEW_ZETA

# ################# COMPUTING THE ELBO #################

def elbo_x(data, prior, hyper, mom, par, remove_self_loops=True):

    if remove_self_loops:
        NON_EDGES = data['NON_X']
    else:
        NON_EDGES = 1.0 - data['X']
    # We use one line from update_z
    ALPHA_diff = sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])
    BETA_diff = sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA'])

    # We use the einsums from update_pi (and add ALPHA_diff, BETA_diff)
    str_sum = 'km,kmiq,kmjr,kij,mqr->'
    P1 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], data['X'], ALPHA_diff)
    P2 = np.einsum(str_sum, mom['MU'], mom['TAU'], mom['TAU'], NON_EDGES, BETA_diff)

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

    P1 = np.einsum('km,kmiq,mq->', np.ones(mom['MU'].shape), mom['TAU'], NU_diff)

    P2 = np.einsum('km,kmiq->', np.ones(mom['MU'].shape), sp.xlogy(mom['TAU'], mom['TAU']))

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
