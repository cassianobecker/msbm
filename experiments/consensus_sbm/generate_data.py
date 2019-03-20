# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np
sys.path.insert(0,'../..')
import generate_msbm as gn

# ################ INITIALIZATION FUNCTIONS ###############

def main():

	for detec in np.arange(0.8,1.25,0.05):
		print("generating networks with SNR:{}".format(detec))
		data, par = gn.create_msbm(
		Q= 4, N = 120, M = 1, K = 100,
		dii = 36.0, dij = 2.0,
		SNR = detec,
		path_data = 'data',
		fname = 'consensus_{:3.0f}'.format(100*detec),
		verbose = False)
	sys.exit()

if __name__ == '__main__':
    main()
