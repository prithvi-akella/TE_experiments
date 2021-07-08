import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
import pdb

class UCBOptimizer:
	def __init__(self, objective, bounds, B, R=1, delta=0.05, n_restarts = 0,
		n_init_samples = 5, tolerance = 0.05, length_scale = 1, constraints = None):
		# Objective here encodes the objective function to be optimized
		# Bounds indicates the bounds over which objective is to be optimized.
		# This algorithm will assume that the bounding region is hyper-rectangular
		# as this assumption can be made trivially (if Objective is to be optimized
		# over a compact, non-rectangular space, and if Objective is continuous over
		# said space, then define a continuous, surjective map from a larger hyper-
		# rectangular space to the compact space in question.  The resulting composite
		# function is a continuous objective function which still optimizes the original).

		# Objective should meet the criteria that objective(x) = y, where x is an 1 x N
		# numpy array and y is a float.

		# Bounds should be an Nx2 numpy array, where N is the dimensionality of the
		# constraint space.  The first column contains the lower bounds, the second the
		# upper bound.

		# B is the presumed upper bound on the RHKS norm for the Objective in question

		# R is the presumed upper bound on the variance of the measurement noise (should
		# there not be measurement noise, R will default to 1)

		# 1-delta is the required confidence, i.e. at termination the result will hold
		# with probability 1-delta (should the bounding values be correct)

		# n_restarts is the number of times the GPR hyperparameters will be re-initialized when
		# fitting (larger n_restarts indicates longer time to convergence, default value = 0 implying
		# that the hyperparameters will not be optimized.  In general, the hyperparameters needn't be
		# optimized as the algorithm will use the universal Matern Kernel.)

		# n_init_samples is the number of samples the algorithm will start off with, the default will be 5,
		# and they will be randomly sampled from the feasible, hyper-rectangular space.  For problems
		# of a high dimension, seeding with a larger number of initial samples will likely expedite the
		# solution process, though it will lead to large runtimes for GPR regression in later stages (as
		# the number of samples explodes).

		# tolerance prescribes the required tolerance within which we would like the optimal value
		# to lie

		# length_scale: if the length_scale of the objective function is known apriori, then please input it
		# (for use in an RBF kernel), else it will be initialized to 1 (for which the kernel is universal)

		# constraint will be initialized to None unless otherwise populated.  We presume constraints is
		# a list of constraint functions for the optimization problem at hand, each of which has the same
		# evaluation scheme, i.e.: constraints[i](x) = some float.  Here, i is the i-th constraint, and
		# x is a 1xN numpy array.  Additionally, we assume wlog that each constraint function is to be kept
		# negative, i.e. the constraints are constraint[i](x) <= 0.

		self.objective = objective
		self.bounds = bounds
		self.beta = []
		self.B = B
		self.R = R
		self.delta = delta
		self.mu = []                      # Initializing fields to contain the mean function and
		self.sigma = []                   # covariance function respectively.
		self.dimension = bounds.shape[0]  # Dimensionality of the feasible space
		self.X_sample = None              # Instantiating a variable to keep track of all sampled, X values
		self.Y_sample = None              # Instantiating a variable to keep track of all sampled, Y values
		self.cmax =  -1e6                 # Instantiating a variable to keep track of the current best value
		self.best_sample = None           # Instantiating a variable to keep track of the current best sample
		self.n_init_samples = n_init_samples
		self.n_restarts = n_restarts
		self.tol = tolerance
		self.max_val = []
		self.term_sigma = 0
		self.UCB_sample_pt = 0
		self.constraints = constraints

	def check_constraints(self,x):
		if self.constraints == None:
			return True
		else:
			flag = True
			for constraint in self.constraints:
				flag = flag and (constraint(x) <=0)
				if flag == False: break
			return flag

	def initialize(self, sample = None, observations = None, init_flag = False):
		# Initialize a starting set of samples and their y-values.  Initial sample size is set to 5

		if init_flag == False:
			sample_flag = False
			while not sample_flag:
				self.X_sample = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
					size=(self.n_init_samples,self.dimension))
				met_constraints_flags = []
				for i in range(self.n_init_samples):
					met_constraints_flags.append(self.check_constraints(self.X_sample[i,:].reshape(1,-1)))
				sample_flag = sample_flag or all(met_constraints_flags)

			self.Y_sample = np.zeros((self.n_init_samples,1))
			for i in range(self.n_init_samples):
				self.Y_sample[i,0] = self.objective(self.X_sample[i,:].reshape(1,-1))
				if self.Y_sample[i] > self.cmax:
					self.cmax = self.Y_sample[i,0]
					self.best_sample = self.X_sample[i,:].reshape(1,-1)
		elif sample is not None:
			self.X_sample = sample
			self.Y_sample = observations
			self.cmax = np.max(self.Y_sample)
			observation_list = observations.reshape(-1,).tolist()
			max_index = observation_list.index(max(observation_list))
			self.best_sample = self.X_sample[max_index,:].reshape(1,-1)
		else:
			error('The initialization flag was set to False and a prior sample set was not provided')


	def UCB(self, x):
		# Calculate the Upper Confidence Bound for a value, x, based on the data-set, (x_sample, y_sample).
		# gpr is the Regressed Gaussian Process defining the current mean and standard deviation.

		# Returning the UCB based on the choice of beta.
		if self.check_constraints(x.reshape(1,-1)):
			return self.mu(x) + self.beta*self.sigma(x)
		else:

			return -100

	def propose_location(self, opt_restarts = 20):
		# Propose the next location to sample (identify the sample point that maximizes the UCB)
		# This is a nonlinear optimization problem and will require a number of restarts
		# The number of restarts will be set to min(N_samples+1,25) to expedite computation

		min_value = 1e6      # Keeping track of the current minimum value (min of negation of UCB)
		min_x = None         # Keeping track of best sample point so far

		def min_obj(x):
			return -self.UCB(x=x)


		# Iterate through opt_restarts IP methods to determine the maximizer of the UCB over the
		# hyper-rectangle identified by bounds.
		if self.constraints is not None:
			x0_array = np.array([])
			init_state_count = 0
			while init_state_count <= opt_restarts:
				sample = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size = (1,self.dimension))
				if self.check_constraints(sample):
					x0_array = np.vstack((x0_array,sample)) if x0_array.shape[0]>0 else sample
					init_state_count += 1
		else:
			x0_array = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(opt_restarts, self.dimension))


		for x0 in x0_array:
			res = minimize(min_obj, x0 = x0, bounds = self.bounds, method='L-BFGS-B')
			if res.fun < min_value:
				min_value = res.fun
				min_x = res.x

		# Output the minimizer (which is the maximizer of the UCB as we're minimizing -UCB)
		return min_x.reshape(1,-1), min_value

	def find_closest(self,x,indeces):
		'''
		Finds the nearest sample point in the sampled x array to the point x provided that also
		satisfies the constraints
		'''
		length_list = [None for i in range(self.X_sample.shape[0])]
		for i in range(self.X_sample.shape[0]):
			length_list[i] = np.linalg.norm(self.X_sample[i,:].reshape(1,-1)-x)

		if self.constraints is not None:
			found_nearest = False
			while not found_nearest:
				minindex = length_list.index(min(length_list))
				if minindex in indeces:
					length_list[minindex] = max(length_list)
				else:
					found_nearest = True

		distance = length_list[minindex]
		return minindex,distance


	def optimize(self):
		# Initialize the nominal Matern Kernel (known to be universal for any parameters)
		kernel = RBF(1.0)
		self.kernel = kernel
		F = 100
		t = 1
		missed_constraints = 0
		indeces = []

		while F >= self.tol:

			Kn = kernel(self.X_sample)
			eta = 2/t
			Kinv = np.linalg.inv(Kn+eta*np.identity(self.X_sample.shape[0]))
			self.Kinv = Kinv
			self.mu = lambda x: np.dot(np.dot(kernel(x.reshape(1,-1), self.X_sample).reshape(1,-1),
				Kinv),self.Y_sample)[0,0]
			self.sigma = lambda x: (kernel(x.reshape(1,-1)) - np.dot(np.dot(kernel(x.reshape(1,-1), self.X_sample).reshape(1,-1), Kinv),kernel(x.reshape(1,-1), self.X_sample).reshape(-1,1)))[0,0]

			innersqrt = np.linalg.det((1+2/t)*np.identity(self.X_sample.shape[0]) + kernel(self.X_sample))
			self.beta = self.B + self.R*math.sqrt(2*math.log(math.sqrt(innersqrt)/self.delta))
			new_x, min_val = self.propose_location(opt_restarts = self.X_sample.shape[0]*2)
			if not self.check_constraints(new_x):
				pdb.set_trace()

			F = 2*self.beta*self.sigma(new_x)
			self.term_sigma = self.sigma(new_x)
			new_y = self.objective(new_x)

			self.UCB_sample_pt = new_x
			self.UCB_val = -min_val

			if not self.check_constraints(new_x):
				missed_constraints += 1
				indeces.append(self.X_sample.shape[0])

			if new_y > self.cmax:
				self.cmax = new_y
				self.best_sample = new_x

			if F >= self.tol:
				self.X_sample = np.vstack((self.X_sample, new_x))
				self.Y_sample = np.vstack((self.Y_sample, new_y))

			print('Finshed with iteration %d'%t)
			print('UCB value at that point: %.5f'%self.UCB_val)
			print('Current best value: {}'.format(self.cmax))
			print('Sample that yielded best value: {}'.format(self.best_sample))
			print('Current F value: %.5f'%F)
			print('Required tolerance: %.5f'%self.tol)
			print('')
			t+=1

		print('Assumed maximum value: %.3f'%(-min_val))
		print('beta at termination: %.3f'%self.beta)
		print('Final variance at termination: %.3f'%self.term_sigma)
		print('Total number of times samples taken that did not satisfy constraints: %d'%missed_constraints)
		print('Indeces in X_sample where the offending samples were taken {}'.format(indeces))
