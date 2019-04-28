# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np

sys.path.insert(0, '../..')
import generate_msbm as gn


# ################ INITIALIZATION FUNCTIONS ###############

def main():
    for detec in np.arange(0.7, 1.4, 0.05):
        print("generating networks with SNR:{}".format(detec))
        data = gn.create_msbm(
            Q=3, N=120, M=1, K=128,
            c = 18,
            SNR= detec,
            path_data='data',
            fname='consensus_{:03.0f}'.format(100 * detec),
            verbose=True)
    sys.exit()


if __name__ == '__main__':
    main()
