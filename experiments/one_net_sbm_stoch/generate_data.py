# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np

sys.path.insert(0, '../..')
import generate_multistrataSBM as gn


# ################ INITIALIZATION FUNCTIONS ###############

def main():
    data=  gn.create_multistrata(
        Q= 4, N = 1500, K = 1, M=1,
        pii = [0.0302], c = 12,
        path_data = 'data',
        fname = 'one_net',
        verbose = True)
    sys.exit()


if __name__ == '__main__':
    main()
