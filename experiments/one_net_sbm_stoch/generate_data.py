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
    for i in range(50):
        j = int(i/10)*0.005
        data= gn.create_multistrata(
            Q= 4, N = 600, K = 1, M=1,
            pii = [0.1 + j], c = 20,
            path_data = 'data',
            fname = 'net{:02d}'.format(i),
            verbose = True)
    sys.exit()


if __name__ == '__main__':
    main()
