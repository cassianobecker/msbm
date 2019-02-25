import numpy as np
import numpy.random as npr
import sys
import pickle
import time
import pdb
from random import randint
import scipy as sp
from scipy.stats import beta
from scipy import optimize

def getavgDeg(gamma, pi,n):
	"""
	Obtain the average degree for an n-node network with community
	weights gamma and connectivity matrix pi
	Parameters
	----------
	gamma : double array 
		probability vector representing community weights
	pi : double array
		vector of probabilities representing inter connection probabilities
	n : integer
		number of nodes
	"""
	#We construct the matrix whose columns are the community profiles
	PQ = np.matmul(np.diag(gamma[0]),pi)
	comDegs = np.sum(PQ*np.log(n),axis=0)
	#Now take the weighted average
	avgDeg =  np.sum(gamma[0]*comDegs)

	return avgDeg

def getSNR(gamma, pi):
	"""
	Criterion for detectability on the general SBM for community weights gamma
	and connectivity matrix pi
	Parameters
	----------
	gamma : double array 
		probability vector representing community weights
	pi : double array
		vector of probabilities representing inter connection probabilities
	"""
	#We construct the matrix whose columns are the community profiles
	PQ = np.matmul(np.diag(gamma[0]),pi)
	snr = 500
	com_i = 0
	com_j = 0
	#Iterate over all different pairs of columns
	for i in range(0,PQ.shape[1]-1):
		for j in range(0,PQ.shape[1]-1):
			if i != j:
				prof_i = PQ[:,i]
				prof_j = PQ[:,j]
				new_snr = CHDiv(prof_i,prof_j)
				if new_snr <= snr:
					snr = new_snr
					com_i = i
					com_j = j
	return snr, com_i, com_j 

def CHDiv(theta_i,theta_j):
	"""
	Chernoff-Hellinger divergence of two community profiles
	detectability depends on whether the CHDiv is greater than 1.
	This is an f-divergence and a generalization of relative-entropy.
	Parameters
	----------
	theta_i : double array 
		profile for community i
	theta_j : double array
		profile for community j
	"""
	#Obtain the f-divergence associated with t
	def fdiv(t):
		return -np.sum(t*theta_i + (1-t)*theta_j - (theta_i**t)*(theta_j**(1-t)))

	#Minimize over t
	res = optimize.minimize(fdiv,(0.5),bounds= ((0,1),))
	chdiv = -fdiv(res.x)
	return chdiv

def accept_prob(e,eprime,T):
	"""
	accept_prob acceptance probability for every step of
	the simulated annealing. We accept the new state whenever
	it is a lower energy state, otherwise we accept with a probability
	that decays exponentially in proportion to the difference of energies
	and is affected by a temperature parameter. 
	Parameters
	----------
	e : double 
		energy of current state
	eprime : double
		energy of proposed state
	T : double
		current temperature parameter
	"""
	if eprime < e:
		p = 1
	else:
		p = np.exp(-(eprime - e)/T)
	return p


# ############# SAMPLING FUNCTIONS #######################
# We sample gamma and pi from Dirichlet and Beta distributions
# respectively by specifying means and variances. Optionally, we
# can constraint on a sample Dt f-divergence, which will determine
# asymptotic detectability. For the constrained sampling we use simulated annealing.


def get_gamma_pi(
	dii, dij,vdir = 0.015, 
	Q = 4, sampling = 'unconstrained', SNR=None,
	maxIter = 100, tol = 1e-07, n = 100, verbose = False):
	"""
	get_gamma_pi a function that samples gamma (the community
	weights or community importance) from a homogeneous Dirichlet
	distribution (which has mean, a uniform distribution)
	and pi (the connection probabilities), by specifying
	means and variances. Optionally, we can constraint the sample to 
	have a target Dt f-divergence (E. Abbe and C. Sandon. 2015)
	which will determine asymptotic detectability. For the constrained
	sampling we use simulated annealing.
	Parameters
	----------
	Q : int 
		the number of communities. Defaults to 4
	sampling : str
		the type of sampling to be made. The options are
		'deterministic', 'unconstrained' and 'constrained'
		which correspond to deterministic annealing 
	vdir : double
		variance of the homogeneous dirichlet distribution
	dii : double
		average degree whithin community, it's assumed to grow logarithmically.
	dij : double
		average degree across community, it's assumed to grow logarithmically.
	SNR : double
		target signal to noise ratio for constrained sampling
	maxIter : int 
		maximum number of iterations for the simulated annealing
	n : int
		the number of nodes. Defaults to 100. It's used to produce a logarithmic degree
		interconnection matrix pi. 
	verbose : boolean
		set to false to run the function quietly.
	"""
	#Obtain the alphas with target var and assuming a uniform mean
	if sampling not in {'constrained','unconstrained'}:
		sys.exit("Unsupported sampling type")

	alpha = ((Q-1)/(vdir*Q**2)+1)/Q
	gamma = npr.dirichlet(np.repeat(alpha, Q),1)
	
	#Beta parameters alpha_ii and beta_ii
	mii = np.double(dii)/n
	vii = (np.double(dii)/10)/(n)

	alpha_ii = ((mii**2)*(1-mii) - mii*vii)/vii
	beta_ii  = (1/vii)*(mii*(1-mii) - vii)*(1-mii)
	#Beta parameters alpha_ij and beta_ij
	mij = np.double(dij)/n
	vij = (np.double(dij)/10)/(n)
	alpha_ij = ((mij**2)*(1-mij) - mij*vij)/vij
	beta_ij  = (1/vij)*(mij*(1-mij) - vij)*(1-mij)

	pi = np.zeros((Q,Q))
	for i in range(Q):
		for j in range(Q):
			if i == j:
				pi[i,j] = npr.beta(alpha_ii,beta_ii)/2

			if i < j:
				pi[i,j] = npr.beta(alpha_ij,beta_ij)

	pi = pi + np.transpose(pi)
	if sampling == 'unconstrained':
		return gamma, pi

	if verbose == True:
		print('Beggining constrained sampling with:{:d} iterations'.format(maxIter))
		print('---------------------------------------------------')
	T_init= 1.01
	#Normalize the average degree of the network to its expected value
	targetDeg = (np.double(dii) + (Q-1)*np.double(dij))/Q
	avgDeg = getavgDeg(gamma,pi * (n/np.log(n)),n)
	pi = pi * targetDeg/avgDeg
	if SNR is None:
		sys.exit("Unspecified SNR. The detectability threshold is SNR = 1")
	counter = 0
	iteration = 0
	pi_constant = pi * (n/np.log(n))
	cSNR, com_i, com_j  = getSNR(gamma,pi_constant)

	e = (SNR - cSNR)**2
	while counter <= maxIter and e > tol:
		avgDeg =  getavgDeg(gamma,pi * (n/np.log(n)),n)
		if counter % 25 == 0 and verbose == True:
			print('Iter:{:d}, objSNR: {:03f}, current_SNR: {:03f}, Energy: {:08f}, avgDeg: {:03f}'.format(counter,SNR,cSNR,e,avgDeg))
		counter += 1
		iteration += 1
		#We restart the simulated annealing every 10000 iterations
		if counter % 20000 == 0:
			if verbose == True:
				time.sleep(1.5)
				print('Restarting Simulated Annealing')
			gamma = npr.dirichlet(np.repeat(alpha, Q),1)
			pi = np.zeros((Q,Q))
			for i in range(Q):
				for j in range(Q):
					if i == j:
						pi[i,j] = npr.beta(alpha_ii,beta_ii)/2

					if i < j:
						pi[i,j] = npr.beta(alpha_ij,beta_ij)

			pi = pi + np.transpose(pi)			
			T_init= 1.01
			avgDeg = getavgDeg(gamma,pi * (n/np.log(n)),n)
			pi = pi * targetDeg/avgDeg
			iteration = 1
		#End the restarting if
		pi_prime = pi
		gamma_prime = gamma
		#We do the local step and compute the energy of that state
		#With probability .5 we change gamma with probability .5 we change pi
		#We do a random step where the average degree of the network is preserved
		p_or_g = npr.binomial(1, .5, 1)
		if p_or_g == 1:
			xx = npr.binomial(1, .5, 1)[0]
			ind_i = xx*com_i + (1-xx)*com_j
			ind_j = randint(0,Q-1)
			#We take a weighted average of a new beta sample and the current one, randomly sampling from 
			if ind_i == ind_j:
				new_value = npr.beta(alpha_ii,beta_ii)
			else:
				new_value = npr.beta(alpha_ij,beta_ij)
			pi_prime[ind_i,ind_j] = 0.99*pi_prime[ind_i,ind_j] + 0.01*new_value
			pi_prime[ind_j,ind_i] = 0.99*pi_prime[ind_j,ind_i] + 0.01*new_value	
		#And for gamma
		else:
			gamma_prime = 0.99*gamma_prime + 0.01*npr.dirichlet(np.repeat(alpha, Q),1)

		#We renormalize to preserve the average degree
		avgDeg_prime = getavgDeg(gamma_prime,pi_prime * (n/np.log(n)),n)
		pi_prime = pi_prime * (targetDeg/avgDeg_prime)

		#Chernoff-Hellinger Divergence must be computed on a constant matrix pi, before the n/log(n) factor
		pi_prime_constant = pi_prime * (n/np.log(n))
		cSNR_prime, com_i_prime, com_j_prime = getSNR(gamma_prime, pi_prime_constant)

		eprime = (SNR - cSNR_prime)**2
		#We obtain the acceptance probability
		p = accept_prob(e,eprime,T_init/(iteration))
		# if(counter > 5000) and (counter % 25 == 0):
		# 	pdb.set_trace()
		if npr.binomial(1, p, 1) == 1:
			if(eprime > e) and verbose==True:
				print("Jumping to state with higher Energy")
			e = eprime
			gamma = gamma_prime
			pi = pi_prime
			cSNR = cSNR_prime
			com_i = com_i_prime
			com_j = com_j_prime
			pi_constant = pi * (n/np.log(n))
	if verbose == True:
		print('Finished simulated annealing after:{:d} iterations '.format(counter-1))
		print('Resulting SNR is:{:04f}, target was {:04f} '.format(cSNR,SNR))
	return gamma, pi


def get_rho():

	rho = np.array([0.2, 0.5, 0.7, 0.3])

	return rho/np.sum(rho)


def sample_Y(rho, K):

	Y = npr.multinomial(1, rho, K)

	return Y.astype(float)


def sample_Z(gamma, N):

	Z = npr.multinomial(1, gamma, N)

	return Z.astype(float)

def sample_X_und(Pi, Z):

	N = Z.shape[0]
	X = np.zeros((N, N))
	for i in range(N):
			zi = find_row(Z[i, :])
			for j in range(i, N):
				zj = find_row(Z[j, :])
				X[i, j] = npr.binomial(1, Pi[zi, zj])
				X[j, i] = X[i, j]

	return X.astype(float)


# ############### MODEL CREATION ################


def create_msbm(Q, N, M, K):

	print('---- Creating MSBM mode with N = {:d} and K = {:d} ------'
		  .format(N, K))

	RHO = get_rho()

	Y = sample_Y(RHO, K)

	GAMMA = np.zeros((M, Q))
	for m in range(M):
		GAMMA[m, :] = get_gamma(m)

	PI = np.zeros((M, Q, Q))
	for m in range(M):
		PI[m, :] = get_pi(m)

	Z = np.zeros((K, N, Q))
	X = np.zeros((K, N, N))
	for k in range(K):
		m = find_row(Y[k, :])
		Z[k, :] = sample_Z(GAMMA[m, :], N)
		X[k, :] = sample_X_und(PI[m, :], Z[k, :])

	par = dict()
	par['Q'] = Q
	par['N'] = N
	par['M'] = M
	par['K'] = K

	data = dict()
	data['RHO'] = RHO
	data['Y'] = Y
	data['GAMMA'] = GAMMA
	data['PI'] = PI
	data['Z'] = Z
	data['X'] = X

	return data, par


def find_row(x):

	return np.nonzero(x)[0][0]

# ############### MAIN PROGRAM #####################


def main():
	gamma, pi = get_gamma_pi(
	vdir = 0.0065, dii = 36.0, dij= 4.0,  
	Q = 3, sampling = 'constrained', SNR = 0.6,
	maxIter = 60000, tol = 1e-5, n = 100, verbose= True)
	SystemExit(0)
	# initialize file names
	# if len(sys.argv) < 2:
	#	 path_data = 'data'
	#	 fname = 'msbm1'
	#	 data_file_url = path_data + '/' + fname + '.pickle'

	# else:
	#	 data_file_url = sys.argv[1]

	# # number of classes
	# Q = 4
	# # number of nodes
	# N = 150
	# # number of models
	# M = 4
	# # number of networks
	# K = 80

	# # sample model
	# data, par = create_msbm(Q, N, M, K)

	# # save to file
	# print('Saving file to {:s} ... '.format(data_file_url), end='')
	# pickle.dump({'data': data, 'par': par}, open(data_file_url, 'wb'))
	# print('saved.')


if __name__ == '__main__':

	main()
