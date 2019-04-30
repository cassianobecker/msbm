import pdb
import updates_msbm_vi as msbm
#import updates_msbm2_vi as msbm
from util import *

# ################### MAIN INFERENCE PROGRAM #####################


def get_default_parameters(par):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if 'kappas' not in par.keys():
        par['kappas'] = sigmoid(np.linspace(0.5, par['MAX_ITER']/4, par['MAX_ITER']))

    if 'nat_step' not in par.keys():
        par['nat_step'] = 0.5

    par['MAX'] = 1000

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

    if sum(np.diff(elbos['all'])<0) > 9:
        stop = True
        reason = 'STOPPED (ELBO oscillated 10 times)'

    if t == par['MAX_ITER'] - 1:
        stop = True
        reason = 'STOPPED (maximum number of iterations = {:d} achieved).'.format(par['MAX_ITER'])

    return stop, reason


def print_header(data, hyper, par):

    print('##############################################################')
    print('                RUNNING ' + str.upper(par['ALG']) + ' FOR MSBM        ')
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


def infer(data, prior, hyper, mom, par):
    #ADD NON_X to the Data (non edges without self loops)
    par = get_default_parameters(par)

    print_header(data, hyper, par)

    elbos = dict()

    for t in range(par['MAX']):

        par['kappa'] = par['kappas'][t]

        elbos = msbm.compute_elbos(data, prior, hyper, mom, par, elbos)

        print_status(t, data, mom, par, elbos)

        stop, reason = check_stopping(t, par, elbos)

        if stop:
            print(reason)
            return mom, elbos

        # ####################### CAVI IMPLEMENTATION ########################

        if par['ALG'] == 'cavi':

            ALPHA, BETA = msbm.update_Pi(data, prior, hyper, mom, par)
            mom['ALPHA'] = ALPHA
            mom['BETA'] = BETA

            NU = msbm.update_gamma(data, prior, hyper, mom, par)
            mom['NU'] = NU

            for _ in range(1):
                LOG_TAU = msbm.update_Z(data, prior, hyper, mom, par)
                mom['LOG_TAU'] = LOG_TAU
                mom['TAU'] = msbm.TAU_from_LOG_TAU(mom, par)

            LOG_MU = msbm.update_Y(data, prior, hyper, mom, par)
            mom['LOG_MU'] = LOG_MU
            mom['MU'] = msbm.par_from_mom_MU(mom, par)

            ZETA = msbm.update_rho(data, prior, hyper, mom, par)
            mom['ZETA'] = ZETA

        # ##################### NATGRAD IMPLEMENTATION #######################

        if par['ALG'] == 'natgrad':

            mom_new = dict()

            ALPHA, BETA = msbm.update_Pi(data, prior, hyper, mom, par)
            mom_new['ALPHA'] = (1.0 - par['nat_step']) * mom['ALPHA'] + par['nat_step'] * ALPHA
            mom_new['BETA'] = (1.0 - par['nat_step']) * mom['BETA'] + par['nat_step'] * BETA

            NU = msbm.update_gamma(data, prior, hyper, mom, par)
            mom_new['NU'] = (1.0 - par['nat_step']) * mom['NU'] + par['nat_step'] * NU

            LOG_TAU = msbm.update_Z(data, prior, hyper, mom, par)
            mom_new['LOG_TAU'] = (1.0 - par['nat_step']) * mom['LOG_TAU'] + par['nat_step'] * LOG_TAU
            mom_new['TAU'] = msbm.TAU_from_LOG_TAU(mom_new, par)

            LOG_MU = msbm.update_Y(data, prior, hyper, mom, par)
            mom_new['LOG_MU'] = (1.0 - par['nat_step']) * mom['LOG_MU'] + (par['nat_step']) * LOG_MU
            mom_new['MU'] = msbm.par_from_mom_MU(mom_new, par)

            ZETA = msbm.update_rho(data, prior, hyper, mom, par)
            mom_new['ZETA'] = (1.0 - par['nat_step']) * mom['ZETA'] + par['nat_step'] * ZETA

            mom = mom_new

    return mom, elbos
