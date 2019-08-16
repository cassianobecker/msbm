import pdb
import updates_msbm_vi_iter_stoch as msbm
import numpy as np
from util import *
import sys
# ################### MAIN INFERENCE PROGRAM #####################


def get_default_parameters(par):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if 'kappas' not in par.keys():
        par['kappas'] = sigmoid(np.linspace(0.5, par['MAX_ITER']/4, par['MAX_ITER']))

    if 'nat_step_rate' not in par.keys():
        par['nat_step_rate'] = 0.9

    if 'MAX_ITER' not in par.keys():
        par['MAX_ITER'] = 1000

    return par


def get_relative_elbo(elbos):

    if len(elbos['all']) > 1:
        relative_elbo = ((elbos['all'][-1] - elbos['all'][-2]) / abs(elbos['all'][-2]))
    else:
        relative_elbo = abs(elbos['all'][-1])

    return relative_elbo


def check_stopping(t, par, elbos):

    stop = False
    reason = ''
    relative_elbo = get_relative_elbo(elbos)

    if abs(relative_elbo) < par['TOL_ELBO']:
        stop = True
        reason = 'STOPPED (ELBO tolerance {:1.4e} achieved).'.format(par['TOL_ELBO'])

    if t == par['MAX_ITER'] - 1:
        stop = True
        reason = 'STOPPED (maximum number of iterations = {:d} achieved).'.format(par['MAX_ITER'])

    return stop, reason


def print_header(data, hyper, par):

    print('##############################################################')
    print('                RUNNING STOCHASTIC NATGRAD (strat) FOR MSBM        ')
    print('                K = {:d}, M = {:d}, N = {:d}, Q = {:}'.
          format(data['K'], hyper['M'], data['N'], hyper['Q']))
    print('##############################################################')


def print_status(t, data, mom, par, elbos):
    ari_Y = adj_rand(mom['MU'], data['Y'])
    mari_Z = np.mean(adj_rand_Z(mom, data))
    mentro_Z = np.mean(get_entropy_Z(mom))
    mentro_Y = np.mean(get_entropy_Y(mom))
    relative_elbo = get_relative_elbo(elbos)

    if 'Y' in data.keys():
        print(
            'Iter: {:3d} of {:3d} (kap = {:.2f}) -- elbo: {:+.5e}, | ari(Y): {:+.3f} |'
            ' aari(Z): {:+.3f} | d_elbo: {:+.4e} | aent(Y): {:+.5f} | aent(Z): {:+.5f}'.format(
                t + 1, par['MAX_ITER'], par['kappa'], elbos['all'][-1], ari_Y, mari_Z, relative_elbo, mentro_Y, mentro_Z))
    else:
        print(
            'Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.5e}'.format(
                t + 1, par['MAX_ITER'], par['kappa'], elbos['all'][-1]))


def peek_TAU( tensor ):
    return tensor[0,0, range(15), :]

def infer(data, prior, hyper, mom, par, verbose = True):
    #ADD NON_X to the Data (non edges without self loops)
    par = get_default_parameters(par)

    if verbose:
        print_header(data, hyper, par)

    elbos = dict()

    #select m so that on average the sets of non-links have the same size as the set of links
    ##Get avg. degree of first network
    avg_deg = np.mean(np.sum(data['X'][0,:,:], axis= 0))
    ##Divide average number of non-links by avg_deg
    m = int((data['N'] - avg_deg)/avg_deg)
    data['strata'] = gen_stratified_sets(data['X'], m)

    for t in range(par['MAX_ITER']):

        par['kappa'] = par['kappas'][t]

        if t%200 == 0 or t%200 == 1:
            elbos = msbm.compute_elbos(data, prior, hyper, mom, par, elbos)

            if verbose:
                print_status(t, data, mom, par, elbos)

            stop, reason = check_stopping(t, par, elbos)

            if stop:
                if verbose:
                    print(reason)
                return mom, elbos

        # ####################### CAVI IMPLEMENTATION ########################

        if par['ALG'] == 'cavi':
            sys.exit("There is no CAVI with stochastic variational inference. Try step size of 1, and no sampling.")

        # ##################### NATGRAD IMPLEMENTATION #######################

        if par['ALG'] == 'natgrad':

            step = (3 + t)**(-par['nat_step_rate'])

            ALPHA, BETA = msbm.update_Pi(data, prior, hyper, mom, par)
            mom['ALPHA'] = (1.0 - step) * mom['ALPHA'] + step * ALPHA
            mom['BETA'] = (1.0 - step) * mom['BETA'] + step * BETA
            
            NU = msbm.update_gamma(data, prior, hyper, mom, par)
            mom['NU'] = (1.0 - step) * mom['NU'] + step * NU

            # LOG_MU = msbm.update_Y(data, prior, hyper, mom, par)
            # mom['LOG_MU'] = (1.0 - step) * mom['LOG_MU'] + (step) * LOG_MU
            # mom['MU'] = msbm.par_from_mom_MU(mom, par)
            # print("Updated Mu")

            # ZETA = msbm.update_rho(data, prior, hyper, mom, par)
            # mom['ZETA'] = (1.0 - step) * mom['ZETA'] + step * ZETA
            LOG_TAU = msbm.update_Z(data, prior, hyper, mom, par)
            mom['LOG_TAU'] = (1.0 - step) * mom['LOG_TAU'] + step * LOG_TAU
            mom['TAU'] = msbm.TAU_from_LOG_TAU(mom, par)

    return mom, elbos

