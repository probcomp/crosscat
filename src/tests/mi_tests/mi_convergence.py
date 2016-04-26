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

import matplotlib
matplotlib.use('Agg')

import numpy
import pylab as pl
import crosscat.tests.mutual_information_test_utils as mitu
import crosscat.utils.inference_utils as iu
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State

from scipy.stats import norm as norm
from scipy.stats import pearsonr as pearsonr

from sklearn.metrics import mutual_info_score

import random
import time
import argparse
import math

import pdb

def sample_from_discrete_joint(fxy,num_cols):
    r = random.random()
    s = 0
    i = 0
    for p in numpy.nditer(fxy):
        s += p
        if s > r:
            col = i % num_cols
            row = math.floor(i/num_cols)
            return row, col
        i += 1

    print("index not found")
    assert(false)

def log_linspace(a,b,n):
    return numpy.exp(numpy.linspace(numpy.log(a),numpy.log(b),n))

def gen_correlated_data( n, rho ):
    T = numpy.random.multivariate_normal([0,0],[[1,rho],[rho,1]],n)
    true_MI = -.5*numpy.log(1-rho*rho);
    external_mi = mitu.mi(T[:,0],T[:,1],base=math.e)
    return T, true_MI, external_mi

def gen_correlated_data_discrete( n, r, SEED=0):
    K = 10

    X = list(range(K))
    Y = list(range(K))

    if r < .05:
        r = .05

    # define joint distribution
    std = K/(6.0*r)
    fxy = numpy.zeros((K,K))
    for row in range(K):
        mu = float(row)
        for col in range(K):
            fxy[row,col] = norm.pdf(float(col),loc=mu,scale=std)

    fxy = fxy/numpy.sum(fxy)

    # marginals
    fx = numpy.sum(fxy, axis=0)
    fx /= numpy.sum(fx)
    fy = numpy.sum(fxy, axis=1)
    fy /= numpy.sum(fy)

    true_MI = 0
    for x in X:
        for y in Y:
            true_MI += fxy[x,y]*numpy.log(fxy[x,y]/(fx[x]*fy[y]))

    external_mi = mutual_info_score(None,None,contengency=fxy)

    # sample data from joint distribution
    T = numpy.zeros((n,2))
    for i in range(n):
        x, y = sample_from_discrete_joint(fxy,K)
        T[i,0] = x
        T[i,1] = y


    return T, true_MI, external_mi

def run_test(args):

    rho = args.rho
    num_times = args.num_times
    min_num_rows = args.min_num_rows
    max_num_rows = args.max_num_rows
    n_grid = args.n_grid
    filename = args.filename
    discrete = args.discrete

    num_samples = []
    for ns in log_linspace(min_num_rows,max_num_rows,n_grid).tolist():
        num_samples.append(int(ns))

    variances = []

    burn_in = 200

    MIs = numpy.zeros( (num_times, len(num_samples)) )

    mi_diff = numpy.zeros( (len(num_samples), num_times) )

    if not discrete:
        T, true_mi, external_mi =  gen_correlated_data(num_samples[-1], rho)
        cctypes = ['continuous']*2
    else:
        T, true_mi, external_mi =  gen_correlated_data_discrete(num_samples[-1], rho)
        cctypes = ['multinomial']*2

    data_subs = []

    get_next_seed = lambda: random.randint(1, 2**31 - 1)

    n_index = 0
    for n in num_samples:
        T_sub = numpy.copy(T[0:n-1,:])
        
        data = []

        data_subs.append(T_sub)

        print("%i: " % n)
        for t in range(num_times):
            M_c = du.gen_M_c_from_T(T_sub,cctypes)
            state = State.p_State(M_c, T_sub)
            state.transition(n_steps=burn_in)
            X_D = state.get_X_D()
            X_L = state.get_X_L()

            MI, Linfoot = iu.mutual_information(M_c, [X_L], [X_D], [(0,1)],
                    get_next_seed, n_samples=5000)

            mi_diff[n_index,t] = true_mi-MI[0][0]

            print("\t%i TRUE: %e, EST: %e " % (t, true_mi, MI[0][0]) )

            MIs[t,n_index] = MI[0][0]

        n_index += 1


    if discrete:
        dtype_str = "discrete"
    else:
        dtype_str = "continuous"

    basefilename = filename + str(int(time.time()))
    figname = basefilename + ".png"
    datname = basefilename + "_DATA.png"

    pl.figure

    # plot data
    # pl.subplot(1,2,1)
    pl.figure(tight_layout=True,figsize=(len(data_subs)*4,4))
    i = 0
    for T_s in data_subs:
        pl.subplot(1,len(data_subs),i+1)
        num_rows = num_samples[i]
        if discrete:
            heatmap, xedges, yedges = numpy.histogram2d(T_s[:,0], T_s[:,1], bins=10)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            pl.imshow(heatmap, extent=extent, interpolation="nearest")
        else:
            pl.scatter(T_s[:,0], T_s[:,1], alpha=.3, s=81)
        pl.title('#r: '+str(num_rows))

        i += 1

    pl.suptitle("data for rho: %1.2f (%s)" % (rho, dtype_str) )

    pl.savefig(datname)
    pl.clf()

    pl.figure(tight_layout=True,figsize=(5,4))
    # plot convergence
    # pl.subplot(1,2,2)
    # standard deviation
    stderr = numpy.std(MIs,axis=0)#/(float(num_times)**.5)
    mean = numpy.mean(MIs,axis=0)
    pl.errorbar(num_samples,mean,yerr=stderr,c='blue')
    pl.plot(num_samples, mean, c="blue", alpha=.8, label='mean MI')
    pl.plot(num_samples, [true_mi]*len(num_samples), color='red', alpha=.8, label='true MI');
    pl.plot(num_samples, [external_mi]*len(num_samples), color=(0,.5,.5), alpha=.8, label='external MI');
    pl.title('convergence')
    pl.xlabel('#rows in X (log)')
    pl.ylabel('CrossCat MI - true MI')

    pl.legend(loc=0,prop={'size':8})
    pl.gca().set_xscale('log')

    # save output
    pl.title("convergence rho: %1.2f (%s)" % (rho, dtype_str) )

    pl.savefig(figname)

if __name__ == "__main__":
    # python mi_convergence.py --num_times 10 --max_num_rows 1000 --max_num_samples 1000 --n_grid 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=float, default=.75)
    parser.add_argument('--num_times', type=int, default=20)
    parser.add_argument('--min_num_rows', type=int, default=100)
    parser.add_argument('--max_num_rows', type=int, default=10000)
    parser.add_argument('--n_grid', type=int, default=10)
    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--filename', type=str, default="mi_convergence")

    args = parser.parse_args()

    argsin = {
        'rho'             : args.rho,
        'num_times'       : args.num_times,
        'min_num_rows'    : args.min_num_rows,
        'max_num_rows'    : args.max_num_rows,
        'n_grid'          : args.n_grid,
        'discrete'        : args.discrete,
        'filename'        : args.filename,
    }

    run_test(args)
