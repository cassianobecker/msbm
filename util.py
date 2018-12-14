import sklearn.metrics as skm
import numpy as np


# ################# AUXILIARY FUNCTIONS #################

def find_col(idc):

    col = [np.nonzero(idc[i, :])[0][0] for i in range(idc.shape[0])]

    return col


def adj_rand(tau, Z):

    rand_index = skm.adjusted_rand_score(find_col(Z), np.argmax(tau, axis=1))

    return rand_index


def adj_rand_Z(mom, data):

    ms = np.argmax(mom['MU'], axis=1)
    arZ = [adj_rand(mom['TAU'][k, m, :], data['Z'][k, :]) for k, m in enumerate(ms)]

    return arZ


def replicate_array(X, d, k):

    for i in range(k):
        X = np.expand_dims(X, axis=d)

    return X
