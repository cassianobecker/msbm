import msbm
import plot
from util import *

# ################### MAIN CAVI PROGRAM #####################


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def infer(mom, data, prior, par, alg='cavi'):

    print('##############################################################')
    print('                RUNNING ' + str.upper(alg) + ' FOR MSBM        ')
    print('                K = {:d}, M = {:d}, N = {:d}, Q = {:}'.
          format(par['K'], par['M'], par['N'], par['Q']))
    print('##############################################################')

    T = par['MAX_ITER']
    lbs = np.array(0)
    kappas = sigmoid(np.linspace(-3, 10, T))
    elbos = dict()

    for t in range(T):

        par['kappa'] = kappas[t]

        elbos, elbo = msbm.compute_elbos(elbos, mom, data, prior, par)
        plot.plot_elbos(elbos, par)

        lbs = np.append(lbs, elbo)

        if 'Y' in data.keys():
            print('Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.5e},  adj. rand index Y: {:+.3f}'
                  .format(t + 1, T, par['kappa'], elbo, adj_rand(mom['MU'], data['Y'])))
        else:
            print('Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.5e}'.format(t+1, T, par['kappa'], elbo))

        # ####################### CAVI IMPLEMENTATION ########################

        if alg == 'cavi':

            ALPHA, BETA = msbm.update_Pi(mom, data, prior, par)
            mom['ALPHA'] = ALPHA
            mom['BETA'] = BETA

            LOG_TAU = msbm.update_Z(mom, data, prior, par)
            mom['LOG_TAU'] = LOG_TAU
            mom['TAU'] = msbm.par_from_mom_TAU(mom, par)

            NU = msbm.update_gamma(mom, data, prior, par)
            mom['NU'] = NU

            LOG_MU = msbm.update_Y(mom, data, prior, par)
            mom['LOG_MU'] = LOG_MU
            mom['MU'] = msbm.par_from_mom_MU(mom, par)

            ZETA = msbm.update_rho(mom, data, prior, par)
            mom['ZETA'] = ZETA

        # ##################### NATGRAD IMPLEMENTATION #######################

        if alg == 'natgrad':

            s_t1 = 0.2
            s_t2 = 1.0
            mom_new = dict()

            ALPHA, BETA = msbm.update_Pi(mom, data, prior, par)
            mom_new['ALPHA'] = (1.0 - s_t2)*mom['ALPHA'] + s_t2*ALPHA
            mom_new['BETA'] = (1.0 - s_t2)*mom['BETA'] + s_t2*BETA

            LOG_TAU = msbm.update_Z(mom, data, prior, par)
            mom_new['LOG_TAU'] = (1.0 - s_t2)*mom['LOG_TAU'] + s_t2*LOG_TAU
            mom_new['TAU'] = msbm.par_from_mom_TAU(mom_new, par)

            NU = msbm.update_gamma(mom, data, prior, par)
            mom_new['NU'] = (1.0 - s_t1)*mom['NU'] + s_t1*NU

            LOG_MU = msbm.update_Y(mom, data, prior, par)
            mom_new['LOG_MU'] = (1.0 - s_t1)*mom['LOG_MU'] + s_t1*LOG_MU
            mom_new['MU'] = msbm.par_from_mom_MU(mom_new, par)

            ZETA = msbm.update_rho(mom, data, prior, par)
            mom_new['ZETA'] = (1.0 - s_t1)*mom['ZETA'] + s_t1*ZETA

            mom = mom_new

    print('\nFinished (maximum number of iterations).')

    return mom, lbs
