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
from __future__ import print_function
import argparse
import random
import tempfile
import time
import sys
from collections import Counter

import numpy
import pylab

import crosscat.tests.enumerate_utils as eu
import crosscat.tests.plot_utils as pu
import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du

import crosscat.cython_code.State as State

def get_next_seed(rng, max_val=32767):
    return rng.randint(max_val)


def run_test(seed, n=1000, d_type='continuous', observed=False):
    if d_type == 'continuous':
        run_test_continuous(n, observed, seed)
    elif d_type == 'multinomial':
        run_test_multinomial(n, observed, seed)

def generate_multinomial_data(seed, n_cols, n_rows, n_views):
    rng = numpy.random.RandomState(seed)

    cols_to_views = [0 for _ in range(n_cols)]
    rows_in_views_to_cols = []
    for view in range(n_views):
        partition = eu.CRP(n_rows, 2.0, get_next_seed(rng))
        random.shuffle(partition, rng.uniform)
        rows_in_views_to_cols.append(partition)

    # generate the data
    data = numpy.zeros((n_rows,n_cols),dtype=float)
    for col in range(n_cols):
        view = cols_to_views[col]
        for row in range(n_rows):
            cluster = rows_in_views_to_cols[view][row]
            data[row,col] = cluster

    T = data.tolist()
    M_r = du.gen_M_r_from_T(T)
    M_c = du.gen_M_c_from_T(T)

    T, M_c = du.convert_columns_to_multinomial(T, M_c, list(range(n_cols)))

    return T, M_r, M_c

def run_test_continuous(n, observed, seed):
    rng = numpy.random.RandomState(seed)

    n_rows = 40
    n_cols = 40

    if observed:
        query_row = 10
    else:
        query_row = n_rows

    query_column = 1

    Q = [(query_row, query_column)]

    # do the test with multinomial data
    T, M_r, M_c= du.gen_factorial_data_objects(get_next_seed(rng),2,2,n_rows,1)

    state = State.p_State(M_c, T)

    T_array = numpy.array(T)

    X_L = state.get_X_L()
    X_D = state.get_X_D()

    Y = [] # no constraints

    # pull n samples
    gns = lambda: get_next_seed(rng)
    samples = su.simple_predictive_sample(M_c, X_L, X_D, Y, Q, gns, n=n)

    X_array = numpy.sort(numpy.array(samples))

    std_X = numpy.std(X_array)
    mean_X = numpy.mean(X_array)

    # filter out extreme values
    X_filter_low = numpy.nonzero(X_array < mean_X-2.*std_X)[0]
    X_filter_high = numpy.nonzero(X_array > mean_X+2.*std_X)[0]
    X_filter = numpy.hstack((X_filter_low, X_filter_high))
    X_array = numpy.delete(X_array, X_filter)

    # sort for area calculation later on
    X_array = numpy.sort(X_array)

    X = X_array.tolist()

    # build the queries
    Qs = [];
    for x in X:
        Qtmp = (query_row, query_column, x)
        Qs.append(Qtmp)

    # get pdf values
    densities = numpy.exp(su.simple_predictive_probability(M_c, X_L, X_D, Y, Qs))

    # test that the area under Ps2 and pdfs is about 1 
    # calculated using the trapezoid rule
    area_density = 0;
    for i in range(len(X)-1):
        area_density += (X[i+1]-X[i])*(densities[i+1]+densities[i])/2.0

    print("Area of PDF (should be close to, but not greater than, 1): " + str(area_density))
    print("*Note: The area will be less than one because the range (integral) is truncated.")

    pylab.figure(facecolor='white')

    # PLOT: probability vs samples distribution
    # scale all histograms to be valid PDFs (area=1)
    pdf, bins, patches = pylab.hist(X,100,normed=1, histtype='stepfilled',label='samples', alpha=.5, color=[.5,.5,.5])
    pylab.scatter(X,densities, c="red", label="pdf", edgecolor='none')

    pylab.legend(loc='upper left',fontsize='x-small')
    pylab.xlabel('value') 
    pylab.ylabel('frequency/density')
    pylab.title('TEST: PDF (not scaled)')

    pylab.show()
    fd, fig_filename = tempfile.mkstemp(prefix='run_test_continuous_',
            suffix='.png', dir='.')
    pylab.savefig(fig_filename)


def run_test_multinomial(n, observed, seed):
    rng = numpy.random.RandomState(seed)

    n_rows = 40
    n_cols = 40

    if observed:
        query_row = 10
    else:
        query_row = n_rows

    query_column = 1

    Q = [(query_row, query_column)]

    # do the test with multinomial data
    T, M_r, M_c = generate_multinomial_data(get_next_seed(rng),2,n_rows,1)

    state = State.p_State(M_c, T)

    X_L = state.get_X_L()
    X_D = state.get_X_D()

    Y = []

    # pull n samples
    gns = lambda: get_next_seed(rng)
    samples = su.simple_predictive_sample(M_c, X_L, X_D, Y, Q, gns, n=n)
    X_array = numpy.sort(numpy.array(samples))
    X = numpy.unique(X_array)
    X = X.tolist()

    # build the queries
    Qs = [];
    for x in X:
        # Qtmp = (query_row, query_column, x[0])
        Qtmp = (query_row, query_column, x)
        Qs.append(Qtmp)

    # get pdf values
    densities = numpy.exp(su.simple_predictive_probability(M_c, X_L, X_D, Y, Qs))

    print("Sum of densities (should be 1): %f" % (numpy.sum(densities)))

    pylab.clf()

    # PLOT: probability vs samples distribution
    # scale all histograms to be valid PDFs (area=1)
    mbins = numpy.unique(X_array)

    mbins = numpy.append(mbins,max(mbins)+1)

    pdf, bins = numpy.histogram(X_array,mbins)

    pdf = pdf/float(numpy.sum(pdf))
    pylab.bar(mbins[0:-1],pdf,label="samples",alpha=.5)
    pylab.scatter(X,densities, c="red", label="pdf", edgecolor='none')

    pylab.legend(loc='upper left',fontsize='x-small')
    pylab.xlabel('value') 
    pylab.ylabel('frequency/density')
    pylab.title('TEST: PDF (not scaled)')

    pylab.show()

    fd, fig_filename = tempfile.mkstemp(prefix='run_test_multinomial_',
            suffix='.png', dir='.')
    pylab.savefig(fig_filename)

rng = numpy.random.RandomState()


run_test(seed=get_next_seed(rng), n=5000, d_type='continuous', observed=False)
run_test(seed=get_next_seed(rng), n=5000, d_type='continuous', observed=True)
run_test(seed=get_next_seed(rng), n=5000, d_type='multinomial', observed=False)
run_test(seed=get_next_seed(rng), n=5000, d_type='multinomial', observed=True)
