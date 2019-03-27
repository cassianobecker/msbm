import updates_msbm_vi as msbm
import plot
import pdb
from scipy import sparse
from util import *
import init_msbm_vi as im


# ################### MAIN CAVI PROGRAM #####################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def infer(mom, data, prior, par):
    print('##############################################################')
    print('                RUNNING ' + str.upper(alg) + ' FOR MSBM        ')
    print('                K = {:d}, M = {:d}, N = {:d}, Q = {:}'.
          format(data['K'], mom['NU'].shape[0], data['N'], mom['ALPHA'].shape[1]))
    print('##############################################################')

    T = par['MAX_ITER']
    lbs = np.array(0)
    if 'KAPPAS' not in par.keys():
        kappas = sigmoid(np.linspace(-3, 10, T))
    elbos = dict()
    alg = par['ALG']
    #TO DO: Implement stopping criterion 
    for t in range(T):

        par['kappa'] = kappas[t]

        elbos = msbm.compute_elbos(mom, data, prior, par, elbos)
        #Unclear what this does, but it requires the key elbos0 which was deleted
        #plot.plot_elbos(elbos, par)

        ari_Y = adj_rand(mom['MU'], data['Y'])
        #mean adjusted rand index for Z
        mari_Z = np.mean(adj_rand_Z(mom, data))

        lbs = np.append(lbs, elbos['all'][-1])

        if t > 0:
            rel_elbo = ((lbs[-1] - lbs[-2]) / abs(lbs[-2]))
            # print('Relative Elbo: {:1.4e}'.format(rel_elbo))
            if abs(rel_elbo) < par['TOL_ELBO']:
                print('\nFinished (ELBO tolerance {:1.4e} achieved).'.format(par['TOL_ELBO']))
                return mom, lbs
        else:
            rel_elbo = 0.


        if 'Y' in data.keys():
            print(
                'Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.5e}, | ari(Y): {:+.3f} | avg. ari(Z): {:+.3f} | delta elbo: {:+.4e} |'.format(
                    t + 1, T, par['kappa'], elbos['all'][-1], ari_Y, mari_Z, rel_elbo))
        else:
            print(
                'Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.5e}'.format(
                    t + 1, T, par['kappa'], elbos['all'][-1]))

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
            #Step Lengths
            s_t1 = 0.5
            s_t2 = 0.5
            mom_new = dict()

            ALPHA, BETA = msbm.update_Pi(mom, data, prior, par)
            mom_new['ALPHA'] = (1.0 - s_t2) * mom['ALPHA'] + s_t2 * ALPHA
            mom_new['BETA'] = (1.0 - s_t2) * mom['BETA'] + s_t2 * BETA

            LOG_TAU = msbm.update_Z(mom, data, prior, par)
            mom_new['LOG_TAU'] = (1.0 - s_t2) * mom['LOG_TAU'] + s_t2 * LOG_TAU
            mom_new['TAU'] = msbm.par_from_mom_TAU(mom_new, par)

            NU = msbm.update_gamma(mom, data, prior, par)
            mom_new['NU'] = (1.0 - s_t1) * mom['NU'] + s_t1 * NU

            LOG_MU = msbm.update_Y(mom, data, prior, par)
            mom_new['LOG_MU'] = (1.0 - s_t1) * mom['LOG_MU'] + s_t1 * LOG_MU
            mom_new['MU'] = msbm.par_from_mom_MU(mom_new, par)

            ZETA = msbm.update_rho(mom, data, prior, par)
            mom_new['ZETA'] = (1.0 - s_t1) * mom['ZETA'] + s_t1 * ZETA

            mom = mom_new



    print('\nFinished (maximum number of iterations).')

    return mom, lbs
