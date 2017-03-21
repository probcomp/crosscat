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
import crosscat.tests.mutual_information_test_utils as mitu
import crosscat.utils.inference_utils as iu
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State

from scipy.stats import pearsonr as pearsonr
import math
import random


def gen_ring( n, width, SEED=0 ):
	numpy.random.seed(SEED)

	i = 0;

	# uses rejections sampling
	T = numpy.zeros( (n,2) )
	while i < n:
		x = numpy.random.uniform(-1,1)
		y = numpy.random.uniform(-1,1)
		rad = (x**2 + y**2)**.5
		if 1.0-w < rad and rad < 1.0:
			T[i,0] = x
			T[i,1] = y
			i += 1

	mi_est = mitu.mi(T[:,0], T[:,1], base=math.e)

	return T, mi_est


cctypes = ['continuous']*2

get_next_seed = lambda : random.randrange(32000)

widths = [.9, .8, .7, .6, .5 , .4, .3, .2, .1]
n = 500
n_samples = 50
pl.figure()
burn_in = 400

mi_ests = numpy.zeros(len(widths))
	
datas = []

nr = 0
for w in widths:
	T, mi_est = gen_ring( n, w, SEED=get_next_seed())

	datas.append(T)

	print("num_samples: %i, width: %f" % (n, w))

	M_c = du.gen_M_c_from_T(T,cctypes)
	X_Ls = []
	X_Ds = []

	for ns in range(n_samples):
		state = State.p_State(M_c, T)
		state.transition(n_steps=burn_in)
		X_Ds.append(state.get_X_D())
		X_Ls.append(state.get_X_L())
	
	MI, Linfoot = iu.mutual_information(M_c, X_Ls, X_Ds, [(0,1)],
            get_next_seed, n_samples=5000)

	data_d = numpy.transpose(MI)

	if nr == 0:
		data = data_d
	else:
		data = numpy.hstack((data, data_d))

	mi_ests[nr] = mi_est
	
	nr += 1


pl.figure(tight_layout=True,figsize=(len(widths)*4,4))
i = 0;
for T_s in datas:
	pl.subplot(1,len(datas),i+1)
	pl.scatter(T_s[:,0], T_s[:,1], alpha=.3, s=81)
	pl.title('w: '+str(widths[i]))
	pl.xlim([-1,1])
	pl.ylim([-1,1])
	i += 1

pl.savefig("ring_data.png")

pl.figure(tight_layout=True,figsize=(6,4))
pl.boxplot(data)
title = "N=%i (ring)" % (n)
pl.ylabel('MI')
pl.xlabel('ring width')
pl.gca().set_xticklabels(widths)
pl.plot(range(1,len(widths)+1), mi_ests, c="red",label="est MI (kd-tree)")
pl.legend(loc=0,prop={'size':8})
pl.title(title)

pl.savefig("mi_vs_ring.png")
