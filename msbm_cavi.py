import numpy as np
import numpy.random as npr
import sys
import scipy.special as sp
import sklearn.metrics as skm
import pickle

################# CAVI UPDATE FUNCTIONS #########################

def update_Pi(mom, data, prior, par):

    strsum = 'km,kij,kmiq,kmjr->mqr'

    NEW_ALPHA = par['kappa']*(prior['ALPHA_0'] +
                       np.einsum(strsum, mom['MU'], data['X'],
                                 mom['TAU'], mom['TAU']) - 1.0) + 1.0

    NEW_BETA = par['kappa']*(prior['BETA_0'] +
                      np.einsum(strsum, mom['MU'], 1.0 - data['X'],
                                mom['TAU'], mom['TAU']) - 1.0) + 1.0

    return NEW_ALPHA, NEW_BETA


def update_Z(mom, data, prior, par):

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))

    S1 = np.einsum('km,mq,i->kmiq', mom['MU'], NUdiff, np.ones(par['N']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(par['N']), np.ones(par['N']), np.ones(par['K']))

    P1 = np.einsum('kij,mqr->mqrijk',
                   data['X'],
                   sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    S2 = np.einsum('km,kmjr,mqrijk->kmiq', mom['MU'], mom['TAU'], P1 + P2)

    NEW_TAU = (S1 + S2)

    NEW_TAU = np.exp(par['kappa']*(NEW_TAU - np.expand_dims(np.max(NEW_TAU, axis=3), axis=3)))

    NEW_TAU = NEW_TAU / np.expand_dims(np.sum(NEW_TAU, axis=3), axis=3)

    return NEW_TAU


def update_Y(mom, data, prior, par):

    ZETAdiff = sp.psi(mom['ZETA']) - sum(sp.psi(mom['ZETA']))

    S1 = np.einsum('m,k->km', ZETAdiff, np.ones(par['K']))

    P1 = np.einsum('kij,mqr->mqrijk', data['X'], sp.psi(mom['ALPHA']) - sp.psi(mom['BETA']))

    P2 = np.einsum('mqr,i,j,k->mqrijk',
                   sp.psi(mom['BETA']) - sp.psi(mom['ALPHA'] + mom['BETA']),
                   np.ones(par['N']), np.ones(par['N']), np.ones(par['K']))

    S2 = np.einsum('kmiq,kmjr,mqrijk->km', mom['TAU'], mom['TAU'], P1 + P2)

    NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))

    S3 = np.einsum('kmiq,mq->km', mom['TAU'], NUdiff)

    NEW_MU = (S1 + S2 + S3)

    NEW_MU = np.exp(par['kappa']*(NEW_MU - np.expand_dims(np.max(NEW_MU, axis=1), axis=1)))

    NEW_MU = NEW_MU / np.expand_dims(np.sum(NEW_MU, axis=1), axis=1)

    return NEW_MU


def update_gamma(mom, data, prior, par):

    NEW_NU = par['kappa']*(prior['NU_0'] +
                           np.einsum('km,kmiq->mq', mom['MU'], mom['TAU']) - 1.0) + 1.0

    return NEW_NU


def update_rho(mom, data, prior, par):

    NEW_ZETA = par['kappa']*(prior['ZETA_0'] +
                             np.einsum('km->m', mom['MU']) - 1.0) + 1.0

    return NEW_ZETA

################## COMPUTING THE ELBO #################
def elbo_x(mom,data,prior,par):
	#We use one line from update_z
	ALPHAdiff = sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])
	BETAdiff  = sp.psi(mom['BETA'])  - sp.psi(mom['ALPHA'] + mom['BETA'])
	#We use the einsums from update_pi (and add ALPHAdiff, BETAdiff)
	strsum = 'km,kmiq,kmjr,kij,mqr->'
	P1 = np.einsum(strsum, mom['MU'],mom['TAU'], mom['TAU'], data['X'], ALPHAdiff)
	P2 = np.einsum(strsum, mom['MU'],mom['TAU'], mom['TAU'], 1.0 -data['X'], BETAdiff)
	lb_x = P1 + P2
	return lb_x

def elbo_gamma(mom, data, prior, par):
	#We use NUdiff from update_z
	NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))
	lb_gamma = np.einsum( 'mq->',(prior['NU_0']- mom['NU'])*NUdiff)
	#Add the \Gamma terms (Not in the updates)
	gammasum_nu   = sum(sp.gammaln(np.einsum('mq->m', mom['NU'])))
	sumgamma_nu   = np.einsum('mq->', sp.gammaln(mom['NU']))
	gammasum_nu0 = sum(sp.gammaln(np.einsum('mq->m', np.ones(mom['NU'].shape)*prior['NU_0'])))
	sumgamma_nu0 = np.einsum('mq->', sp.gammaln( np.ones(mom['NU'].shape)*prior['NU_0'] ))
	lb_gamma = lb_gamma + gammasum_nu0 - sumgamma_nu0 - gammasum_nu + sumgamma_nu
	return lb_gamma

def elbo_rho(mom,data,prior,par):
	#We use ZETAdiff from update_y
	ZETAdiff = sp.psi(mom['ZETA']) - sum(sp.psi(mom['ZETA']))
	lb_rho   = sum((prior['ZETA_0']- mom['ZETA'])*ZETAdiff)
	#We add gamma terms (not in any update)
	gammasum_zeta   = sp.gammaln(sum(mom['ZETA']))
	sumgamma_zeta   = sum(sp.gammaln(mom['ZETA']))
	gammasum_zeta0  = sp.gammaln(sum( prior['ZETA_0']*np.ones(mom['ZETA'].shape)))
	sumgamma_zeta0  = sum(sp.gammaln( prior['ZETA_0']*np.ones(mom['ZETA'].shape)))
	lb_rho = lb_rho + gammasum_zeta0 - sumgamma_zeta0 - gammasum_zeta + sumgamma_zeta
	return lb_rho

def elbo_pi(mom,data,prior,par):
	#We use one line from update_Z
	lb_alpha = np.einsum('mqr->',(prior['ALPHA_0'] - mom['ALPHA'])*(sp.psi(mom['ALPHA']) - sp.psi(mom['ALPHA'] + mom['BETA'])))
	lb_beta  = np.einsum('mqr->',(prior['BETA_0']  -  mom['BETA'])*(sp.psi(mom['BETA'])  - sp.psi(mom['ALPHA'] + mom['BETA'])))
	#We add the gamma terms
	gammasum_ab = np.einsum('mqr->',sp.gammaln(mom['ALPHA']+mom['BETA']))
	sumgamma_ab = np.einsum('mqr->',sp.gammaln(mom['ALPHA']) + sp.gammaln(mom['BETA']))
	gammasum_ab0 = np.einsum('mqr->',sp.gammaln( prior['ALPHA_0']*np.ones(mom['ALPHA'].shape) + prior['BETA_0']*np.ones(mom['BETA'].shape)))
	sumgamma_ab0 = np.einsum('mqr->',sp.gammaln(prior['ALPHA_0']*np.ones(mom['ALPHA'].shape)) + sp.gammaln(prior['BETA_0']*np.ones(mom['BETA'].shape)) )    
	lb_pi = lb_alpha + lb_beta + gammasum_ab0 - sumgamma_ab0 - gammasum_ab + sumgamma_ab
	return lb_pi

def elbo_y(mom,data,prior,par):
	#We use ZETAdiff from update_y
	ZETAdiff = sp.psi(mom['ZETA']) - sum(sp.psi(mom['ZETA']))
	lb_y = np.einsum('km,m->',mom['MU'],ZETAdiff) - np.einsum( 'km->', sp.xlogy(mom['MU'],mom['MU']))
	return lb_y

def elbo_z(mom,data,prior,par):
	#We use NUdiff from update_z
	NUdiff = sp.psi(mom['NU']) - sp.psi(np.einsum('ij,k->ik', mom['NU'], np.ones(par['Q'])))
	P1     = np.einsum( 'km,kmiq,mq->',mom['MU'],mom['TAU'],NUdiff)
	P2     = np.einsum( 'km,kmiq->', mom['MU'],sp.xlogy(mom['TAU'],mom['TAU']))
	lb_z   = P1 - P2
	return lb_z

def compute_elbo(mom,data,prior,par):
    elbo = dict()
    elbo['x'] = elbo_x(mom,data,prior,par)
    elbo['rho']  = elbo_rho(mom,data,prior,par)
    elbo['pi'] =   elbo_pi(mom,data,prior,par)
    elbo['gamma']= elbo_gamma(mom,data,prior,par)
    elbo['y'] = elbo_y(mom, data,prior,par)
    elbo['z'] = elbo_z(mom,data,prior,par)
    return sum(elbo.values())
################## AUXILIARY FUNCTIONS #################


def find_col(idc):

    col = [np.nonzero(idc[i, :])[0][0] for i in range(idc.shape[0])]

    return col


def adj_rand(tau, Z):

    rand_index = skm.adjusted_rand_score(find_col(Z), np.argmax(tau, axis=1))

    return rand_index


################# INITIALIZATION FUNCTIONS ###############

def load_data(data_file_url):

    print('\nLoading data from to {:s} ... '.format(data_file_url), end='')
    loaded = pickle.load(open(data_file_url, 'rb'))
    print('loaded.')

    return loaded['data'], loaded['par']


def init_moments(par):

    npr.seed(123)

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

    TAU = npr.rand(par['K'], par['M'], par['N'], par['Q'])
    for k in range(par['K']):
        for m in range(par['M']):
            TAU[k, m, :] = TAU[k, m, :] / np.expand_dims(np.sum(TAU[k, m, :], axis=1), axis=1)

    mom['TAU'] = TAU

    return mom, prior

#################### MAIN CAVI PROGRAM #####################

def cavi_msbm(mom, data, prior, par):

    print('##############################################################')
    print('------------------- RUNNING CAVI FOR MSBM --------------------')
    print('                K = {:d}, M = {:d}, N = {:d}, Q = {:}'.
          format(par['K'], par['M'], par['N'], par['Q']))
    print('##############################################################')

    T = par['MAX_ITER']
    lbs= np.array(0)
    for t in range(T):

#        par['kappa'] = float((t+10))/float((T+10))
        par['kappa'] = 1.0        
        elbo = compute_elbo(mom,data,prior,par)
        lbs = np.append(lbs,elbo)

        print('Iter: {:3d} of {:3d} (kappa = {:.4f}) --- elbo: {:+.3f}, adj. rand index: {:+.3f}'
              .format(t+1, T, par['kappa'], elbo ,adj_rand(mom['MU'], data['Y'])))

        ALPHA, BETA = update_Pi(mom, data, prior, par)
        mom['ALPHA'] = ALPHA
        mom['BETA'] = BETA
        print('PARTIAL ELBO: {:+.3f}'.format(compute_elbo(mom,data,prior,par)))
        TAU = update_Z(mom, data, prior, par)
        mom['TAU'] = TAU
        print('PARTIAL ELBO: {:+.3f}'.format(compute_elbo(mom,data,prior,par)))
        NU = update_gamma(mom, data, prior, par)
        mom['NU'] = NU
        print('PARTIAL ELBO: {:+.3f}'.format(compute_elbo(mom,data,prior,par)))
        MU = update_Y(mom, data, prior, par)
        mom['MU'] = MU
        print('PARTIAL ELBO: {:+.3f}'.format(compute_elbo(mom,data,prior,par)))
        ZETA = update_rho(mom, data, prior, par)
        mom['ZETA'] = ZETA
        print('PARTIAL ELBO: {:+.3f}'.format(compute_elbo(mom,data,prior,par)))        
        
    print('\nFinished (maximum number of iterations).')

    return mom, lbs


def test_cavi(data_file_url, out_file_url):

    data, par = load_data(data_file_url)

    mom, prior = init_moments(par)

    par['MAX_ITER'] = 15

    results_mom, lbs = cavi_msbm(mom, data, prior, par)

    print('\nSaving results to {:s} ... '.format(out_file_url), end='')
    pickle.dump({'mom': results_mom,'lbs' : lbs}, open(out_file_url, 'wb'))
    print('Saved.')


def main():

    if len(sys.argv) < 3:

        path_data = 'data'
        fname = 'msbm1'
        data_file_url = path_data + '/' + fname + '.pickle'
        out_file_url = path_data + '/' + 'results_' + fname + '.pickle'

    else:
        data_file_url = sys.argv[1]
        out_file_url = sys.argv[2]

    test_cavi(data_file_url, out_file_url)


if __name__ == '__main__':
    main()