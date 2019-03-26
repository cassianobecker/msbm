# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np
sys.path.insert(0,'../..')
import util as ut
import init_msbm_vi as im
import varinf 

def main():

	file_list = os.listdir('data')

	for data_file in file_list:
		#load data
		data, par = ut.load_data('data/' + data_file)
		#initialize moments
		#(TO DO) init_moments should get a "hyper" instead of a par
		# and it's responsibility of the user to provide it
		#hyper = {}/or get it from data
		mom, prior = im.init_moments(par)
    	#set max iterations
		par['MAX_ITER'] = 10
		#(TO DO) infer should need only algorithmic pars
		results_mom, elbo_seq = varinf.infer(mom, data, prior, par, 'cavi')
		print('Saving file to {:s} ... '.format('models/model_' + data_file))
		pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open('models/model_' + data_file, 'wb'))
	sys.exit()

if __name__ == '__main__':
    main()
