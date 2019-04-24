# Experiment that generates several networks (just one) of varying CH-divergence types
# then trains an msbm of a single type. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np

sys.path.insert(0, '../..')
import generate_msbm as gn


# ################ INITIALIZATION FUNCTIONS ###############

def main():
    for detec in [0.5, 0.8, 1, 1.2, 1.5]:
        #we generate 50 networks for each parameter, and store it in a list
        print("generating networks with SNR:{}".format(detec))
        for i in range(6):
            data = gn.create_msbm(
                Q=3, N=300, M=1, K=1,
                c= 15,
                SNR= [detec],
                path_data='data',
                fname='detectability_CH{:03.0f}_{}'.format(100 * detec,i),
                verbose=True)
    #Next, we generate 50 networks for each parameter for N at the detectability threshold
    for N in [100, 400, 1000]:
        print("generating networks with {} nodes".format(N))
        for i in range(6):
            data = gn.create_msbm(
                Q=3, N=N, M=1, K=1,
                c = 15,
                SNR= [1.1],
                path_data='data',
                fname='detectability_N{}_{}'.format(N,i),
                verbose=True)
    #Next, we generate 50 networks for each parameter for N at the detectability threshold
    # for Q in [2, 3, 4]:
    #     print("generating networks with {} nodes".format(N))
    #     for i in range(2):
    #         data = gn.create_msbm(
    #             Q=Q, N=300, M=1, K=1,
    #             dii=46.0, dij=7.0,
    #             SNR=1.2,
    #             path_data='data',
    #             fname='detectability_Q{}_{}'.format(Q, i),
    #             verbose=True)
    sys.exit()


if __name__ == '__main__':
    main()
