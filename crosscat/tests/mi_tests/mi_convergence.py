#
#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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


def gen_correlated_data( n, rho ):
	T = numpy.random.multivariate_normal([0,0],[[1,rho],[rho,1]],n)

	return T

def get_mvnorm_mi( rho ):
	return -.5*numpy.log(1-rho*rho);

num_samples = []
# for ns in numpy.linspace(50,1000,5).tolist():
for ns in numpy.linspace(50,5000,10).tolist():
	num_samples.append(int(ns))

# num_samples = [50,100,250,500]
num_times = 20;

rho = .75

true_MI = [ get_mvnorm_mi(rho) ]*len(num_samples)

variances = []

pl.figure(tight_layout=True,figsize=(10,4))
burn_in = 200

MIs = numpy.zeros( (num_times, len(num_samples)) )

n_index = 0
for n in num_samples:
	
	data = []

	T = gen_correlated_data( n, rho)
	print("%i: " % n)
	for t in range(num_times):
		print("\t%i " % t)
		M_c = du.gen_M_c_from_T(T)
		state = State.p_State(M_c, T)
		state.transition(n_steps=burn_in)
		X_D = state.get_X_D()
		X_L = state.get_X_L()

		MI, Linfoot = iu.mutual_information(M_c, [X_L], [X_D], [(0,1)], n_samples=500)

		MIs[t,n_index] = MI[0][0]

	n_index += 1

	
pl.subplot(1,2,1)
for i in range(len(num_samples)):
	x = [num_samples[i]]*num_times
	y = MIs[:,i]
	pl.scatter(x,y,c='blue',edgecolor='none',s=64,alpha=.25)

stddev = numpy.std(MIs,axis=0)
mean = numpy.mean(MIs,axis=0)

# pdb.set_trace()

pl.plot(num_samples, mean+stddev, c="blue", alpha=.8)
pl.plot(num_samples, mean, c="blue", alpha=.8, label='mean MI')
pl.plot(num_samples, mean-stddev, c="blue", alpha=.8)

pl.plot(num_samples, true_MI, color='red', alpha=.8, label='true MI');
pl.title('convergece')
pl.xlabel('#samples in X')
pl.ylabel('MI')


num_mi_samples = [50,100,200,400,800,1600,3200,6400,12800]
# num_mi_samples = [50,100,200,400,800,1600]
MI_std = numpy.zeros((num_times, len(num_mi_samples)))
n_index = 0
T = gen_correlated_data(5000, rho)
M_c = du.gen_M_c_from_T(T)
state = State.p_State(M_c, T)
state.transition(n_steps=burn_in)
X_D = state.get_X_D()
X_L = state.get_X_L()
for n in num_mi_samples:	
	print("%i: " % n)
	for t in range(num_times):
		print("\t%i " % t)
		MI, Linfoot = iu.mutual_information(M_c, [X_L], [X_D], [(0,1)], n_samples=n)
		MI_std[t,n_index] = MI[0][0]
	n_index += 1

pl.subplot(1,2,2)
pl.plot(num_mi_samples, numpy.std(MI_std,axis=0))
pl.title('stddev of MI with n=%i datatset' % 5000)
pl.xlabel('num MI samples')
pl.ylabel('stddev')

pl.savefig('MI_convergence.png')
