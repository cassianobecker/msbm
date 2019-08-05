"""
Functions for updating the variational parameters for the variational inference
"""
import numpy as np
import scipy.special as sp
import updates_msbm_vi_iter as ul
import pdb
# ################ UPDATE FUNCTIONS #########################


def update_Pi(data, prior, hyper, mom, par, remove_self_loops):

    NEW_ALPHA, NEW_BETA = ul.update_Pi(data, prior, hyper, mom, par, remove_self_loops)

    return NEW_ALPHA, NEW_BETA

# ###########################################################


def update_Z(data, prior, hyper, mom, par, remove_self_loops):

    LOG_TAU = np.zeros((data['K'], data['M'], data['N'], data['Q']))

    for m in range(data['M']):
        for k in range(data['K']):

            for i in range(data['N']):
                for q in range(data['Q']):
                    LOG_TAU[k, m, i, q] = get_log_tau_kmiq(
                        i, q,
                        data['N'], data['Q'], data['X'][k, :, :],
                        mom['TAU'][k, m, :, :],
                        mom['ALPHA'][m, :, :],
                        mom['BETA'][m, :, :],
                        mom['NU'][m, :],
                        mom['MU'][k, m],
                        remove_self_loops)

    NEW_TAU = np.exp(LOG_TAU - np.expand_dims(np.max(LOG_TAU, axis=1), axis=1))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=1), axis=1)

    NEW_LOG_TAU = np.log(NEW_TAU)

    return NEW_LOG_TAU


def get_log_tau_kmiq(i, q, N, Q, Xk, tau_km, a_m, b_m, nu_m, mu_km, remove_self_loops):

    pnuq = sp.psi(nu_m[q])
    psnuq = sp.psi(np.sum(nu_m))
    log_tau_km_x = np.zeros((N, Q))
    log_tau_km_non_x = np.zeros((N, Q))

    for j in range(N):
        for r in range(Q):

            if not (remove_self_loops and j == i):

                x_kij = Xk[i, j]
                tau_kmjr = tau_km[j, r]

                psi_a_mqr = sp.psi(a_m[q, r])
                psi_b_mqr = sp.psi(b_m[q, r])
                psi_ab_mqr = sp.psi(a_m[q, r] + b_m[q, r])

                log_tau_km_x[j, r] = tau_kmjr * (x_kij * (psi_a_mqr - psi_ab_mqr))
                log_tau_km_non_x[j, r] = tau_kmjr * ((1-x_kij)*(psi_b_mqr - psi_ab_mqr))



    log_tau_kmiq = pnuq - psnuq + mu_km*(np.sum(np.sum(log_tau_km_x)) + np.sum(np.sum(log_tau_km_non_x)))

    return log_tau_kmiq


# ###################################################################################

def update_Y(data, prior, hyper, mom, par, remove_self_loops):

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


def get_log_mu_km(m, N, Q, Xk, tau_km, a_m, b_m, zeta, nu_m, remove_self_loops):

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

    # remove this block in comparison to 'updates_msbm_loop' implementation
    # for i in range(N):
    #     for q in range(Q):
    #         tau_kmiq = tau_km[i, q]
    #         psi_nu_mq = sp.psi(nu_m[q])
    #         sum_nu_m = sp.psi(np.sum(nu_m))
    #         log_mu_km = log_mu_km + tau_kmiq * (psi_nu_mq - sum_nu_m)

    return log_mu_km


# ########################################################################

def update_gamma(data, prior, hyper, mom, par):

    NEW_NU = ul.update_gamma(data, prior, hyper, mom, par)

    return NEW_NU


# ########################################################################

def update_rho(data, prior, hyper, mom, par):

    NEW_ZETA = ul.update_rho(data, prior, hyper, mom, par)

    return NEW_ZETA
