#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
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
import numpy
import pylab as pl
import crosscat.utils.inference_utils as iu
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State

from scipy.stats import pearsonr as pearsonr

import random

def get_correlations(T, Q):
	T = numpy.array(T)
	corr = []

	for query in Q:
		r, p = scipy.stats.pearsonr(T[:,Q[0]], T[:,Q[1]])
		corr.append(r)

	return corr

def gen_correlated_data( n, r, SEED=0 ):
	numpy.random.seed(SEED)
	T = numpy.random.multivariate_normal([0,0],[[1,r],[r,1]],n)

	return T

get_next_seed = lambda : random.randrange(32000)

correlations = [.0, .1, .2, .3, .4 , .5, .6, .7, .8, .9, 1.0]
# N = [5, 10, 25, 50]
N = [100]
n_samples = 10
n_data_sets = 3
pl.figure()
burn_in = 200

subplot = 0
for n in N:
	subplot += 1
	nr = 0
	
	for r in correlations:
		for d in range(n_data_sets): # 3 data sets
			#
			T = gen_correlated_data( n, r, SEED=get_next_seed())

			pr, p = pearsonr(T[:,0], T[:,1])

			print "num_samples: %i, R: %f, d: %i. Actual R: %f" % (n, r, d+1, pr)

			M_c = du.gen_M_c_from_T(T)
			X_Ls = []
			X_Ds = []

			for _ in range(n_samples):
				state = State.p_State(M_c, T)
				state.transition(n_steps=burn_in)
				X_Ds.append(state.get_X_D())
				X_Ls.append(state.get_X_L())
			
			MI, Linfoot = iu.mutual_information(M_c, X_Ls, X_Ds, [(0,1)], n_samples=200)

			if d == 0:
				data_d = numpy.transpose(Linfoot)
			else:
				data_d = numpy.vstack((data_d, numpy.transpose(Linfoot)))

		if nr == 0:
			data = data_d
		else:
			data = numpy.hstack((data, data_d))
		
		nr += 1


	pl.subplot(1,1,subplot)
	pl.boxplot(data)
	title = "N=%i" % n
	pl.title(title)

pl.show()
