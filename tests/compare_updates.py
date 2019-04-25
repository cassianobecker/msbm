import updates_msbm_vi_iter
import updates_msbm_vi
import updates_msbm2_vi_iter
import updates_msbm2_vi

import os
import util
import init_msbm_vi as im
import numpy as np
import numpy.random as npr


class TestUpdates:

    def __init__(self, updater_einsum, updater_iter, file_url, remove_self_loops):

        print("\n\n=====================================================================================")
        print("Comparing '{:}' with '{:}' with remove_self_loops={:}"
              .format(updater_einsum.__name__, updater_iter.__name__, remove_self_loops))
        print("=====================================================================================")

        data = util.load_data(file_url)
        self.data = data

        print('')

        K = 8

        data['X'] = data['X'][:K, :, :]

        data['K'] = K
        data['Y'] = data['Y'][:K, :]

        prior = dict()
        prior['ALPHA_0'] = 0.5
        prior['BETA_0'] = 0.5
        prior['NU_0'] = 0.5
        prior['ZETA_0'] = 0.5
        self.prior = prior

        # assigning hyper-parameters from ground truth (cheating)
        hyper = dict()
        hyper['M'] = data['M']
        hyper['Q'] = data['Q']
        self.hyper = hyper

        # initialize moments
        mom = dict()
        npr.seed(1)
        mode = 'random'
        mom['ALPHA'] = im.init_ALPHA(data, hyper, mode)
        mom['BETA'] = im.init_BETA(data, hyper, mode)
        mom['NU'] = im.init_NU(data, hyper, mode)
        mom['ZETA'] = im.init_ZETA(data, hyper, mode)
        mom['MU'] = im.init_MU(data, hyper, mode)
        mom['LOG_MU'] = np.log(mom['MU'])
        mom['TAU'] = im.init_TAU(data, hyper, mode)
        mom['LOG_TAU'] = np.log(mom['TAU'])
        self.mom = mom

        par = dict()
        par['MAX_ITER'] = 1000
        par['TOL_ELBO'] = 1.e-16
        par['ALG'] = 'cavi'
        par['kappa'] = 1.0
        self.par = par

        self.remove_self_loops = remove_self_loops
        self.msbm_einsum = updater_einsum
        self.msbm_iter = updater_iter

    def test_update_Pi(self):
        print('--- Pi ---')

        NEW_ALPHA1, NEW_BETA1 = self.msbm_einsum.update_Pi(self.data, self.prior, self.hyper, self.mom, self.par)
        NEW_ALPHA2, NEW_BETA2 = self.msbm_iter.update_Pi(self.data, self.prior, self.hyper, self.mom, self.par,
                                                         remove_self_loops=self.remove_self_loops)

        self.eval_diff(NEW_ALPHA1, NEW_ALPHA2)
        self.eval_diff(NEW_BETA1, NEW_BETA2)

    def test_update_Z(self):
        print('--- Z ---')

        NEW_LOG_TAU1 = self.msbm_einsum.update_Z(self.data, self.prior, self.hyper, self.mom, self.par)
        NEW_LOG_TAU2 = self.msbm_iter.update_Z(self.data, self.prior, self.hyper, self.mom, self.par,
                                               remove_self_loops=self.remove_self_loops)

        self.eval_diff(NEW_LOG_TAU1, NEW_LOG_TAU2)

    def test_update_Y(self):
        print('--- Y ---')

        NEW_LOG_MU1 = self.msbm_einsum.update_Y(self.data, self.prior, self.hyper, self.mom, self.par)
        NEW_LOG_MU2 = self.msbm_iter.update_Y(self.data, self.prior, self.hyper, self.mom, self.par,
                                              remove_self_loops=self.remove_self_loops)

        self.eval_diff(NEW_LOG_MU1, NEW_LOG_MU2)

    def test_update_gamma(self):
        print('---  Gamma ---')

        NEW_NU1 = self.msbm_einsum.update_gamma(self.data, self.prior, self.hyper, self.mom, self.par)
        NEW_NU2 = self.msbm_iter.update_gamma(self.data, self.prior, self.hyper, self.mom, self.par)

        self.eval_diff(NEW_NU1, NEW_NU2)

    def test_update_rho(self):
        print('---  Rho ---')

        NEW_ZETA1 = self.msbm_einsum.update_rho(self.data, self.prior, self.hyper, self.mom, self.par)
        NEW_ZETA2 = self.msbm_iter.update_rho(self.data, self.prior, self.hyper, self.mom, self.par)

        self.eval_diff(NEW_ZETA1, NEW_ZETA2)

    def test_all(self):
        self.test_update_Pi()
        self.test_update_Z()
        self.test_update_Y()
        self.test_update_gamma()
        self.test_update_rho()

    def eval_diff(self, X1, X2):

        diff = (X1 - X2).ravel()
        print('Mean abs entry-wise error: {:1.3e}'.format(np.mean(np.abs(diff))))
        print('Max abs entry-wise error:  {:1.3e}'.format(np.max(np.abs(diff))))
        print('')

# ###########################################################
# ###########################################################
# ###########################################################

if __name__ == '__main__':

    file_url = os.path.join('..', 'experiments', 'two_prototype', 'data', 'twoprototype_105_250.pickle')

    remove_self_loops = False

    updater_einsum = updates_msbm_vi
    updater_iter = updates_msbm_vi_iter

    runner = TestUpdates(updater_einsum, updater_iter, file_url, remove_self_loops)
    runner.test_all()

    updater_einsum = updates_msbm2_vi
    updater_iter = updates_msbm2_vi_iter

    runner = TestUpdates(updater_einsum, updater_iter, file_url, remove_self_loops)
    runner.test_all()

    remove_self_loops = True

    updater_einsum = updates_msbm_vi
    updater_iter = updates_msbm_vi_iter

    runner = TestUpdates(updater_einsum, updater_iter, file_url, remove_self_loops)
    runner.test_all()

    updater_einsum = updates_msbm2_vi
    updater_iter = updates_msbm2_vi_iter

    runner = TestUpdates(updater_einsum, updater_iter, file_url, remove_self_loops)
    runner.test_all()
