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
    for ind in range(50):
        data = gn.create_msbm(
            Q= 3, N=100, M=2, K=110,
            c = [18],
            SNR= [1.0, 1.2],
            path_data='data',
            fname='twoprototype_{:02}'.format(ind),
            verbose=True)
    sys.exit()


if __name__ == '__main__':
    main()
