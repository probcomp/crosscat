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

# calculates mutual information of a 2 column data set with different correlations
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy
import pylab as pl
import crosscat.utils.inference_utils as iu
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State

from scipy.stats import pearsonr as pearsonr

import random

import pdb

def get_correlations(T, Q):
	T = numpy.array(T)
	corr = []

	for query in Q:
		r, p = scipy.stats.pearsonr(T[:,Q[0]], T[:,Q[1]])
		corr.append(r)

	return corr

def gen_correlated_data( n, r, SEED=0 ):
	rng = numpy.random.RandomState(SEED)
	T = rng.multivariate_normal([0,0],[[1,r],[r,1]],n)

	return T

def gen_correlated_data_discrete( n, r, SEED=0):
	K = 8
	rng = numpy.random.RandomState(SEED)
	T = rng.multivariate_normal([0,0],[[1,r],[r,1]],n)

	min_x = numpy.min(T[:,0])
	min_y = numpy.min(T[:,1])

	T_min_x = numpy.min(T[:,0])
	T_max_x = numpy.max(T[:,0])

	T_min_y = numpy.min(T[:,1])
	T_max_y = numpy.max(T[:,1])

	T_range_x = T_max_x-T_min_x
	T_range_y = T_max_y-T_min_y

	
	T[:,0] -= T_min_x
	T[:,1] -= T_min_y

	T[:,0] /= T_range_x
	T[:,1] /= T_range_y
	T *= K
	
	T = numpy.round(T)

	return T


discrete = True

if discrete:
	cctypes = ['multinomial']*2
else:
	cctypes = ['continuous']*2


get_next_seed = lambda : random.randrange(32000)

correlations = [.0, .1, .2, .3, .4 , .5, .6, .7, .8, .9, 1.0]
n = 250
n_samples = 10
n_data_sets = 3
pl.figure()
burn_in = 200

	
nr = 0
for r in correlations:
	for d in range(n_data_sets): # 3 data sets
		
		if discrete:
			T = gen_correlated_data_discrete( n, r, SEED=get_next_seed())
		else:
			T = gen_correlated_data( n, r, SEED=get_next_seed())

		pr, p = pearsonr(T[:,0], T[:,1])

		print("num_samples: %i, R: %f, d: %i. Actual R: %f" % (n, r, d+1, pr))

		M_c = du.gen_M_c_from_T(T,cctypes)
		X_Ls = []
		X_Ds = []

		for _ in range(n_samples):
			state = State.p_State(M_c, T)
			state.transition(n_steps=burn_in)
			X_Ds.append(state.get_X_D())
			X_Ls.append(state.get_X_L())
		
		MI, Linfoot = iu.mutual_information(M_c, X_Ls, X_Ds, [(0,1)],
                    get_next_seed, n_samples=5000)

		if d == 0:
			data_d = numpy.transpose(Linfoot)
		else:
			data_d = numpy.vstack((data_d, numpy.transpose(Linfoot)))

	if nr == 0:
		data = data_d
	else:
		data = numpy.hstack((data, data_d))
	
	nr += 1


if discrete:
	dt_string = "discrete"
else:
	dt_string = "continuous"

pl.boxplot(data)
title = "N=%i (%s)" % (n,dt_string)
pl.ylabel('Linfoot')
pl.xlabel('data rho')
pl.gca().set_xticklabels(correlations)
pl.plot([1.0, len(correlations)], [0.0, 1.0],c='red')
pl.title(title)

pl.savefig("mi_vs_corr.png")
