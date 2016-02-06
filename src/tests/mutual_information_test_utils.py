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
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State
import random
import numpy

def mi(x,y,k=3,base=2):
    """ Mutual information of x and y
      x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
    """
    x = [[entry] for entry in x]
    y = [[entry] for entry in y]
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
    y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
    points = zip2(x,y)
    #Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
    a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(len(x)) 
    return (-a-b+c+d)/log(base)

def avgdigamma(points,dvec):
    #This part finds number of neighbors in some radius in the marginal space
    #returns expectation value of <psi(nx)>
    N = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    for i in range(N):
        dist = dvec[i]
        #subtlety, we don't include the boundary point, 
        #but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
        avg += digamma(num_points)/N
    return avg

def zip2(*args):
    #zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
    #E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
    return [sum(sublist,[]) for sublist in zip(*args)]


# Generates a num_rows by num_cols array of data with covariance matrix I^{num_cols}*corr
def generate_correlated_data(num_rows, num_cols, means, corr, seed=0):
    assert(corr <= 1 and corr >= 0)
    assert(num_cols == len(means))

    numpy.random.seed(seed=seed)

    mu = numpy.array(means)
    sigma = numpy.ones((num_cols,num_cols),dtype=float)*corr
    for i in range(num_cols):
        sigma[i,i] = 1 
    X = numpy.random.multivariate_normal(mu, sigma, num_rows)

    return X

def generate_correlated_state(num_rows, num_cols, num_views, num_clusters, mean_range, corr, seed=0):
    #

    assert(num_clusters <= num_rows)
    assert(num_views <= num_cols)
    T = numpy.zeros((num_rows, num_cols))

    random.seed(seed)
    numpy.random.seed(seed=seed)
    get_next_seed = lambda : random.randrange(2147483647)

    # generate an assignment of columns to views (uniform)
    cols_to_views = range(num_views)
    view_counts = numpy.ones(num_views, dtype=int)
    for i in range(num_views, num_cols):
        r = random.randrange(num_views)
        cols_to_views.append(r)
        view_counts[r] += 1

    random.shuffle(cols_to_views)

    assert(len(cols_to_views) == num_cols)
    assert(max(cols_to_views) == num_views-1)

    # for each view, generate an assignment of rows to num_clusters
    row_to_clusters = []
    cluster_counts = []
    for view in range(num_views):
        row_to_cluster = range(num_clusters)
        cluster_counts_i = numpy.ones(num_clusters,dtype=int)
        for i in range(num_clusters, num_rows):
            r = random.randrange(num_clusters)
            row_to_cluster.append(r)
            cluster_counts_i[r] += 1

        random.shuffle(row_to_cluster)

        assert(len(row_to_cluster) == num_rows)
        assert(max(row_to_cluster) == num_clusters-1)

        row_to_clusters.append(row_to_cluster)
        cluster_counts.append(cluster_counts_i)

    assert(len(row_to_clusters) == num_views)

    # generate the correlated data
    for view in range(num_views):
        for cluster in range(num_clusters):
            cell_cols = view_counts[view]
            cell_rows = cluster_counts[view][cluster]
            means = numpy.random.uniform(-mean_range/2.0,mean_range/2.0,cell_cols)
            X =  generate_correlated_data(cell_rows, cell_cols, means, corr, seed=get_next_seed())
            # get the indices of the columns in this view
            col_indices = numpy.nonzero(numpy.array(cols_to_views)==view)[0]
            # get the indices of the rows in this view and this cluster
            row_indices = numpy.nonzero(numpy.array(row_to_clusters[view])==cluster)[0]
            # insert the data
            for col in range(cell_cols):
                for row in range(cell_rows):
                    r = row_indices[row]
                    c = col_indices[col]
                    T[r,c] = X[row,col]


    M_c = du.gen_M_c_from_T(T)
    M_r = du.gen_M_r_from_T(T)
    X_L, X_D = generate_X_L_and_X_D(T, M_c, cols_to_views, row_to_clusters, seed=get_next_seed())

    return  T, M_c, M_r, X_L, X_D, cols_to_views

def generate_X_L_and_X_D(T, M_c, cols_to_views, row_to_clusters, seed=0):
    state = State.p_State(M_c, T, SEED=seed)
    X_L = state.get_X_L()

    # insert assigment into X_L (this is not a valid X_L because the counts and 
    # suffstats will be wrong)
    X_L['column_partition']['assignments'] = cols_to_views
    state = State.p_State(M_c, T, X_L=X_L, X_D=row_to_clusters, SEED=seed)

    X_L = state.get_X_L()
    X_D = state.get_X_D()

    return X_L, X_D

