"""
Functions for updating the variational parameters for the variational inference
"""
import numpy as np
import scipy.special as sp

# ################ UPDATE FUNCTIONS #########################


def update_Pi(data, prior, hyper, mom, par, remove_self_loops):

    NEW_ALPHA = np.empty((data['M'], data['Q'], data['Q']))
    NEW_BETA = np.empty((data['M'], data['Q'], data['Q']))

    for m in range(data['M']):
        for q in range(data['Q']):
            for r in range(data['Q']):

                [aqr, bqr] = get_sum_a_b_pi(m, q, r, data, mom, par, prior, remove_self_loops)

                NEW_ALPHA[m, q, r] = prior['ALPHA_0'] + aqr

                NEW_BETA[m, q, r] = prior['BETA_0'] + bqr

    return NEW_ALPHA, NEW_BETA


def get_sum_a_b_pi(m, q, r, data, mom, par, prior, remove_self_loops):

    aqr = 0
    bqr = 0

    for k in range(data['K']):
        for i in range(data['N']):
            for j in range(data['N']):
                if not (remove_self_loops and j == i):

                    aqr = aqr + data['X'][k, i, j]*mom['MU'][k, m] \
                        * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]

                    bqr = bqr + (1 - data['X'][k, i, j])*mom['MU'][k, m] \
                        * mom['TAU'][k, m, i, q] * mom['TAU'][k, m, j, r]

    return aqr, bqr


# ###########################################################

def update_Z(data, prior, hyper, mom, par, remove_self_loops):

    NEW_LOG_TAU = np.empty((data['K'], data['M'], data['N'], data['Q']))
    for m in range(data['M']):
        for k in range(data['K']):

            for i in range(data['N']):
                for q in range(data['Q']):

                    NEW_LOG_TAU[k, m, i, q] = get_log_tau_kmiq(
                        i, q,
                        data['N'], data['Q'], data['X'][k, :, :],
                        mom['TAU'][k, m, :, :],
                        mom['ALPHA'][m, :, :],
                        mom['BETA'][m, :, :],
                        mom['NU'][m, :],
                        mom['MU'][k, m],
                        remove_self_loops)

    return NEW_LOG_TAU


def get_log_tau_kmiq(i, q, N, Q, Xk, tau_km, a_m, b_m, nu_m, mu_km, remove_self_loops):

    pnuq = sp.psi(nu_m[q])
    psnuq = sp.psi(np.sum(nu_m))
    log_tau_kmiq = pnuq - psnuq

    log_tau_km = np.empty((N, Q))

    for j in range(N):
        for r in range(Q):

            if not (remove_self_loops and j == i):

                x_kij = Xk[i, j]
                tau_kmjr = tau_km[j, r]

                psi_a_mqr = sp.psi(a_m[q, r])
                psi_b_mqr = sp.psi(b_m[q, r])
                psi_ab_mqr = sp.psi(a_m[q, r] + b_m[q, r])

                log_tau_km[j, r] = tau_kmjr * (x_kij * (psi_a_mqr - psi_b_mqr) + psi_b_mqr - psi_ab_mqr)

    log_tau_kmiq = mu_km * (log_tau_kmiq + np.sum(np.sum(log_tau_km)))

    return log_tau_kmiq


# ###################################################################################

def update_Y(data, prior, hyper, mom, par, remove_self_loops):

    NEW_LOG_MU = np.empty((data['K'], data['M']))
    NEW_MU = np.empty((data['K'], data['M']))

    for k in range(data['K']):
        for m in range(data['M']):

            NEW_LOG_MU[k, m] = get_log_mu_km(m,
                                             data['N'], data['Q'], data['X'][k, :, :],
                                             mom['TAU'][k, m, :, :],
                                             mom['ALPHA'][m, :, :],
                                             mom['BETA'][m, :, :],
                                             mom['ZETA'],
                                             mom['NU'][m, :],
                                             remove_self_loops)

    new_mu_km = np.empty((data['K'], data['M']))
    for k in range(data['K']):
        Tk = np.max(NEW_LOG_MU[k, :])
        new_mu_km[k, :] = np.exp(NEW_LOG_MU[k, :] - Tk)
        NEW_MU[k, :] = new_mu_km[k, :] / np.sum(new_mu_km[k, :])

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

    for i in range(N):
        for q in range(Q):
            tau_kmiq = tau_km[i, q]
            psi_nu_mq = sp.psi(nu_m[q])
            sum_nu_m = sp.psi(np.sum(nu_m))
            log_mu_km = log_mu_km + tau_kmiq * (psi_nu_mq - sum_nu_m)

    return log_mu_km


# ########################################################################

def update_gamma(data, prior, hyper, mom, par):

    NEW_NU = np.empty((data['M'], data['Q']))

    for m in range(data['M']):
        for q in range(data['Q']):
            nu_mq = 0

            for k in range(data['K']):
                nu_mq = nu_mq + mom['MU'][k, m] * np.sum(mom['TAU'][k, m, :, q])

            NEW_NU[m, q] = prior['NU_0'] + nu_mq

    return NEW_NU


# ########################################################################

def update_rho(data, prior, hyper, mom, par):

    NEW_ZETA = np.empty((data['M']))

    for m in range(data['M']):
        NEW_ZETA[m] = prior['ZETA_0'] + np.sum(mom['MU'][:, m])

    return NEW_ZETA
