#
#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
import crosscat.utils.data_utils as du
import crosscat.utils.sample_utils as su

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext
import crosscat.tests.component_model_extensions.CyclicComponentModel as cycmext

import random
import numpy
import math
import six

# default parameters for 'seeding' random categories
default_data_parameters = dict(
    multinomial=dict(weights=[1.0/5.0]*5),
    continuous=dict(mu=0.0, rho=1.0),
    cyclic=dict(mu=math.pi, kappa=2.0)
    )

get_data_generator = dict(
	multinomial=mcmext.p_MultinomialComponentModel.generate_data_from_parameters,
	continuous=ccmext.p_ContinuousComponentModel.generate_data_from_parameters,
	cyclic=cycmext.p_CyclicComponentModel.generate_data_from_parameters
	)

NaN = float('nan')

has_key = lambda dictionary, key : key in dictionary.keys()

def p_draw(M):
	r = random.random()
	for i in range(len(M)):
		if r < M[i]:
			return i

def add_missing_data_to_column(X, col, proportion):
	"""	Adds NaN entried to propotion of the data X in column col
	"""
	assert proportion >= 0 and proportion <= 1

	for row in range(X.shape[0]):
		if random.random() < proportion:
			X[row,col] = NaN

	return X

def generate_separated_multinomial_weights(A,C):
	"""Generates a set of multinomial weights B, where sum(abs(B-A)) = C
		Inputs:
			A: a list of multinomial weights
			C: A float, 0 <= C <= 1
	"""

	if not isinstance(A, list):
		raise TypeError("A should be a list")

	if not math.fabs(1-sum(A)) < .0000001:
		raise ValueError("A must sum to 1.")

	if C > 1.0 or C < 0:
		raise ValueError("0 <= C <= 1")

	if C == 0.0:
		return A

	idx = [i[0] for i in sorted(enumerate(A), key=lambda x:x[1])]
	A_sum = [A[i] for i in idx]

	A = numpy.array(A)

	A_sum = numpy.cumsum(numpy.array(A_sum))
	
	B = numpy.copy(A)

	t = numpy.nonzero(A_sum >= .5)[0][0]; 

	err_up = idx[:t]
	err_dn = idx[t:]

	upper_bounds = 1.0-B;
	upper_bounds[err_dn] = 0

	lower_bounds = numpy.copy(B);
	lower_bounds[err_up] = 0

	for _ in range(int(C*10.0)):
		# increase a weight
		ups = numpy.nonzero(upper_bounds >= .05)[0]
		move_up = ups[random.randrange(len(ups))]
		B[move_up] += .05
		upper_bounds[move_up] -= .05

		# decrease a weight
		dns = numpy.nonzero(lower_bounds >= .05)[0]
		# if there is no weight to decrease
		if len(dns) == 0:
			# send the lowest weight to zero, normalize and return
			maxdex = numpy.argmin(lower_bounds)
			B[maxdex] = 0
			B /= numpy.sum(B)
			break

		move_down = dns[random.randrange(len(dns))]
		B[move_down] -= .05
		lower_bounds[move_down] -= .05
	
	assert math.fabs(1-numpy.sum(B)) < .0000001
	return B.tolist()

def generate_separated_model_parameters(cctype, C, num_clusters, get_next_seed, distargs=None):
	""" Generates a list of separated component model parameters
	"""
	if cctype == 'continuous':
		# C=1 implies 3 sigma, C=0, implies total overlap, C=.5 implies 1 sigma
		A = 1.7071
		B = .7929
	    # outputs distance in standard deviations that the two clusters should be apart
		d_in_simga = lambda c : A*(c**1.5) + B*c
		rho_to_sigma = lambda rho : (1.0/rho)**.5
		
		# generate the 'seed' component model randomly
		N = 100 # imaginary data
		model = ccmext.p_ContinuousComponentModel.from_parameters(N, gen_seed=get_next_seed())
		params = model.sample_parameters_given_hyper(gen_seed=get_next_seed())
		# track the means and standard deviations

		model_params = [params]
		
		for i in range(0,num_clusters-1):
			params = model.sample_parameters_given_hyper(gen_seed=get_next_seed())
			last_mean = model_params[i]['mu']
			std1 = rho_to_sigma(model_params[i]['rho'])
			std2 = rho_to_sigma(params['rho'])
			sumstd = std1+std2
			push = d_in_simga(C)*sumstd
			params['mu'] = model_params[i]['mu'] + push
			model_params.append(params)

		assert len(model_params) == num_clusters
		random.shuffle(model_params)
		assert len(model_params) == num_clusters
		return model_params

	elif cctype == 'multinomial':
		
		# check the distargs dict	
		if not isinstance(distargs, dict):
			raise TypeError("for cctype 'multinomial' distargs must be a dict")

		try:
			K = distargs['K']
		except KeyError:
			raise KeyError("for cctype 'multinomial' distargs should have key 'K',\
			 the number of categories")

		# generate an inital set of parameters
		# weights = numpy.random.rand(K)
		# weights = weights/numpy.sum(weights)
		weights = numpy.array([1.0/float(K)]*K)
		weights = weights.tolist()

		model_params = [dict(weights=weights)]
		
		for i in range(0,num_clusters-1):
			weights = generate_separated_multinomial_weights(weights,C)
			model_params.append(dict(weights=weights))

		assert len(model_params) == num_clusters
		random.shuffle(model_params)
		assert len(model_params) == num_clusters
		return model_params
	elif cctype == 'cyclic':

		sep = (2.0*math.pi/num_clusters)

		mus = [c*sep for c in range(num_clusters)]
		std = sep/(5.0*C**.75)
		k = 1/(std*std)

		model_params = []
		for c in range(num_clusters):
			model_params.append(dict(mu=mus[c], kappa=k))

		return model_params
	else:
		raise ValueError("Invalid cctype %s." % cctype )


def gen_data(cctypes, n_rows, cols_to_views, cluster_weights, separation, seed=0, distargs=None, return_structure=False):
	"""	Generates a synthetic data.
		Inputs:
			- cctypes: List of strings. Each entry, i, is the cctype of the 
			column i. ex: cctypes = ['continuous','continuous', 'multinomial']
			- n_rows: integer. the number of rows
			- cols_to_views: List of integers. Each entry, i, is the view, v, 
			to which columns i is assigned. v \in [0,...,n_cols-1].
			ex: cols_to_views = [0, 0, 1]
			- cluster_weights: List of lists of floats. A num_views length list
			of list. Each sublist, W, is a list of cluster weights for the 
			view, thus W should always sum to 1.
			ex (two views, first view has 2 clusters, second view has 3 
			clusters):
			cluster_weights = [[.3, .7], [.25, .5, .25]]
			- separation: list of floats. Each entry, i, is the separation, C,
			of the clusters in view i. C \in [0,1] where 0 is no separation and
			1 is well-separated.
			ex (2 views): separation = [ .5, .7]
			- seed: optional
			- distargs: optional (only if continuous). distargs is n_columns
			length list where each entry is either None or a dict appropriate 
			for the cctype in that column. For a normal feature, the entry 
			should be None, for a multinomial feature, the entry should be a 
			dict with the entry K (the number of categories). 
			- return_structure: (bool, optional). Returns also a dict withe the
			data generation structure included. A dict with keys:
				- component_params:  a n_cols length list of lists. Where each 
				list is a set of component model parameters for each cluster in
				the view to which that column belongs
				- cols_to_views: a list assigning each column to a view
				- rows_to_clusters: a n_views length list of list. Each entry,
				rows_to_clusters[v][r] is the cluster to which all rows in 
				columns belonging to view v are assigned
		Returns:
			T, M_c
		Example:
			>>> cctypes = ['continuous','continuous','multinomial','continuous','multinomial']
			>>> disargs = [None, None, dict(K=5), None, dict(K=2)]
			>>> n_rows = 10
			>>> cols_to_views = [0, 0, 1, 1, 2]
			>>> cluster_weights = [[.3, .7],[.5, .5],[.2, .3, .5]]
			>>> separation = [.9, .6, .9]
			>>> T, M_c = gen_data(cctypes, n_rows, cols_to_views, cluster_weights,
				separation, seed=0, distargs=distargs)
	"""

	# check Inputs
	if not isinstance(n_rows, int):
		raise TypeError("n_rows should be an integer")

	if not isinstance(cctypes, list):
		raise TypeError("cctypes should be a list")

	n_cols_cctypes = len(cctypes)
	for cctype in cctypes:
		if not isinstance(cctype, str):
			raise TypeError("cctypes should be a list of strings")

		# NOTE: will have to update when new component models are added
		if cctype not in ['continuous', 'multinomial', 'cyclic']:
			raise ValueError("invalid cctypein cctypes: %s." % cctype)

	if not isinstance(cols_to_views, list):
		raise TypeError("cols_to_views should be a list")

	if len(cols_to_views) != n_cols_cctypes:
		raise ValueError("number of columns in cctypes does not match number\
		 of columns in cols_to_views")

	if min(cols_to_views) != 0:
		raise ValueError("min value of cols_to_views should be 0")

	n_views_cols_to_views = max(cols_to_views) + 1

	set_ctv = set(cols_to_views)
	if len(set_ctv) != n_views_cols_to_views:
		raise ValueError("View indices skipped in cols_to_views")

	# check cluster weights
	if not isinstance(cluster_weights, list):
		raise TypeError("cluster_weights should be a list")

	if n_views_cols_to_views != len(cluster_weights):
		raise ValueError("The number of views in cols_to_views and \
			cluster_weights do not agree.")

	# check each set of weights
	for W in cluster_weights:
		if not isinstance(W, list):
			raise TypeError("cluster_weights should be a list of lists")
		if math.fabs(sum(W)-1.0) > .0000001:
			raise ValueError("each vector of weights should sum to 1")

	if not isinstance(separation, list):
		raise TypeError("separation should be a list")

	if len(separation) != n_views_cols_to_views:
		raise ValueError("number of view in separation and cols_to_views do not agree")

	for c in separation:
		if not isinstance(c, float) or c > 1.0 or c < 0.0:
			raise ValueError("each value in separation should be a float from 0 to 1")

	num_views = len(separation)
	n_cols = len(cols_to_views)

	# check the cctypes vs the distargs
	if distargs is None:
		distargs = [None for i in range(n_cols)]

	if not isinstance(distargs, list):
		raise TypeError("distargs should be a list")

	if len(distargs) != n_cols:
		raise ValueError("distargs should have an entry for each column")

	for i in range(n_cols):
		if cctypes[i] == 'continuous' or cctypes[i] == 'cyclic':
			if distargs[i] is not None:
				raise ValueError("distargs entry for 'continuous' cctype should be None")
		elif cctypes[i] == 'multinomial':
			if not isinstance(distargs[i], dict):
				raise TypeError("ditargs for cctype 'multinomial' should be a dict")
			if len(distargs[i].keys()) != 1:
				raise KeyError("distargs for cctype 'multinomial' should have one key, 'K'")
			if 'K' not in distargs[i].keys():
				raise KeyError("distargs for cctype 'multinomial' should have the key 'K'")
		else:
			raise ValueError("invalid cctypein cctypes: %s." % cctypes[i])

	random.seed(seed)
	numpy.random.seed(seed)

	# Generate the rows to categories partitions (mutlinomial)
	rows_to_clusters = []
	for W in cluster_weights:

		cW = list(W)
		for i in range(1, len(cW)):
			cW[i] += cW[i-1]

		K = len(cW)

		rows_to_clusters_view = list(range(K))
		for r in range(K,n_rows):
			rows_to_clusters_view.append(p_draw(cW))

		random.shuffle(rows_to_clusters_view)
		assert len(rows_to_clusters_view) == n_rows

		rows_to_clusters.append(rows_to_clusters_view)


	get_next_seed = lambda : random.randrange(2147483647)

	# start generating the data
	data_table = numpy.zeros((n_rows, n_cols))
	component_params = []
	for col in range(n_cols):
	
		view = cols_to_views[col]

		# get the number of cluster in view
		num_clusters = len(cluster_weights[view])

		cctype = cctypes[col]

		C = separation[view]

		# generate a set of C-separated component model parameters 
		component_parameters = generate_separated_model_parameters(cctype, C,
			num_clusters, get_next_seed, distargs=distargs[col])

		component_params.append(component_parameters)

		# get the data generation function
		gen = get_data_generator[cctype]
		for row in range(n_rows):
			# get the cluster this 
			cluster = rows_to_clusters[view][row]
			params = component_parameters[cluster]
			x = gen(params, 1, gen_seed=get_next_seed())[0]
			data_table[row,col] = x


	T = data_table.tolist()
	M_c = du.gen_M_c_from_T(T, cctypes=cctypes)

	if return_structure:
		structure = dict()
		structure['component_params'] = component_params
		structure['cols_to_views'] = cols_to_views
		structure['rows_to_clusters'] = rows_to_clusters
		structure['cluster_weights'] = cluster_weights
		return T, M_c, structure
	else:
		return T, M_c

def predictive_columns(M_c, X_L, X_D, columns_list, optional_settings=False, seed=0):
	""" Generates rows of data from the inferred distributions
	Inputs:
		- M_c: crosscat metadata (See documentation)
		- X_L: crosscat metadata (See documentation)
		- X_D: crosscat metadata (See documentation)
		- columns_list: a list of columns to sample
		- optinal_settings: list of dicts of optional arguments. Each column
		  in columns_list should have its own list entry which is either None
		  or a dict with possible keys:
			- missing_data: Proportion missing data
	Returns:
		- a num_rows by len(columns_list) numpy array, where n_rows is the
		original number of rows in the crosscat table. 
	"""
	# supported arguments for optional_settings
	supported_arguments = ['missing_data']

	num_rows = len(X_D[0])
	num_cols = len(M_c['column_metadata'])

	if not isinstance(columns_list, list):
		raise TypeError("columns_list should be a list")

	for col in columns_list:
		if not isinstance(col, int):
			raise TypeError("every entry in columns_list shuold be an integer")
		if col < 0 or col >= num_cols:
			raise ValueError("%i is not a valid column. Should be valid entries\
			 are 0-%i" % (col, num_cols))

	if not isinstance(seed, int):
		raise TypeError("seed should be an int")

	if seed < 0:
		raise ValueError("seed should be positive")

	if optional_settings:
		if not isinstance(optional_settings, list):
			raise TypeError("optional_settings should be a list")

		for col_setting in optional_settings:
			if isinstance(col_setting, dict):
				for key, value in six.iteritems(col_setting):
					if key not in supported_arguments:
						raise KeyError("Invalid key in optional_settings, '%s'" % key)
	else:
		optional_settings = [None]*len(columns_list)

	random.seed(seed)

	X = numpy.zeros((num_rows, len(columns_list)))

	get_next_seed = lambda : random.randrange(2147483647)

	for c in range(len(columns_list)):
		col = columns_list[c]
		for row in range(num_rows):
			X[row,c] = su.simple_predictive_sample(M_c, X_L, X_D, [],
						 [(row,col)], get_next_seed, n=1)[0][0]

		# check if there are optional arguments
		if isinstance(optional_settings[c], dict):
			# missing data argument
			if has_key(optional_settings[c], 'missing_data'):
				proportion = optional_settings[c]['missing_data']
				X = add_missing_data_to_column(X, c, proportion)

	assert X.shape[0] == num_rows
	assert X.shape[1] == len(columns_list)

	return X





