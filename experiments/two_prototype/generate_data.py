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
    data = gn.create_msbm(
        Q=3, N=128, M=2, K=60,
        c = 25,
        SNR= [1.05, 2.5],
        path_data='data',
        fname='twoprototype_105_250',
        verbose=True)
    print("First Prototype:")
    print(data['PI'][0].round(2))
    print("Second Prototype:")
    print(data['PI'][1].round(2))
    sys.exit()


if __name__ == '__main__':
    main()
