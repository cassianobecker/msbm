# Experiment that generates several sets of networks of varying CH-divergence types
# then trains an msbm of a single type in a "consensus" type of way. Then we report the
# average rand_index and average entropy of the z variables, which are indicators of how well 
# the algorithm is learning the true model. 
import os, sys, inspect
import pickle
import numpy as np
sys.path.insert(0,'../..')
import util as ut
import varinf 

def main():

	file_list = os.listdir('data')
	#Need to read in original data + trained moments and elbos
	chd_list = np.array(0)
	ari_list = np.array(0)
	delta_elbo_list = np.array(0)
	entropy_list = np.array(0)
	for data_file in file_list:
		print("Reading data + moments from model: {}".format(data_file))
		#load data
		data, par = ut.load_data('data/' + data_file)
		results_mom, elbo_seq = ut.load_results('data/' + data_file)

		chd_list = np.append(chd_list,par['SNR'])
		#Adjusted rand index
		mari_Z = np.mean(adj_rand_Z(results_mom, data))
		ari_list = np.append(ari_list, mari_Z)
		#Percentage elbo increase
		delta_elbo = (elbos['all'][-1] - elbos['all'][0])/np.abs(elbos['all'][0])
		delta_elbo_list = np.append(delta_elbo_list, delta_elbo)
		#Mean z entropy
		entro = np.mean(get_entropy_Z(results_mom))
		entropy_list = np.append(entropy_list,entro)

		print('Saving file to {:s} ... '.format('models/model_' + data_file))
		pickle.dump({'results_mom': results_mom, 'elbo_seq': elbo_seq}, open('models/model_' + data_file, 'wb'))
	sys.exit()

if __name__ == '__main__':
    main()
