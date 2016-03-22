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
import numpy
import math

from crosscat.utils.general_utils import logsumexp
import crosscat.utils.sample_utils as su

def column_is_bounded_discrete(M_c, col_index):
    return M_c['column_metadata'][col_index]['modeltype'] == 'symmetric_dirichlet_discrete'

def mutual_information_to_linfoot(MI):
    # sometimes esimate code gives MI < 0 which would lead to complex linfoot
    if MI < 0:
        MI = 0
    return (1.0-math.exp(-2.0*MI))**0.5

# return the estimated mutual information for each pair of columns on Q given
# the set of samples in X_Ls and X_Ds. Q is a list of tuples where each tuple
# contains X and Y, the columns to compare.
# Q = [(X_1, Y_1), (X_2, Y_2), ..., (X_n, Y_n)]
# Returns a list of list where each sublist is a set of MI's and Linfoots from
# each crosscat posterior sample.
# See tests/test_mutual_information.py and
# tests/test_mutual_information_vs_correlation.py for useage examples
def mutual_information(M_c, X_Ls, X_Ds, Q, get_next_seed, n_samples=1000):
    #
    assert len(X_Ds) == len(X_Ls)
    n_postertior_samples = len(X_Ds)

    n_cols = len(M_c['column_metadata'])

    MI = []
    Linfoot = []

    for query in Q:
        assert len(query) == 2
        assert query[0] >= 0 and query[0] < n_cols
        assert query[1] >= 0 and query[1] < n_cols

        X = query[0]
        Y = query[1]

        MI_sample = []
        Linfoot_sample = []

        for sample in range(n_postertior_samples):

            X_L = X_Ls[sample]
            X_D = X_Ds[sample]

            # get column data types
            if column_is_bounded_discrete(M_c, X) and column_is_bounded_discrete(M_c, Y):
                MI_s = calculate_MI_bounded_discrete(X, Y, M_c, X_L, X_D)
            else:
                MI_s = estimate_MI_sample(X, Y, M_c, X_L, X_D, get_next_seed, n_samples=n_samples)

            linfoot = mutual_information_to_linfoot(MI_s)

            MI_sample.append(MI_s)

            Linfoot_sample.append(linfoot)

        MI.append(MI_sample)
        Linfoot.append(Linfoot_sample)


    assert len(MI) == len(Q)
    assert len(Linfoot) == len(Q)

    return MI,  Linfoot

def calculate_MI_bounded_discrete(X, Y, M_c, X_L, _X_D):
    get_view_index = lambda which_column: X_L['column_partition']['assignments'][which_column]

    view_X = get_view_index(X)
    view_Y = get_view_index(Y)

    # independent
    if view_X != view_Y:
        return 0.0

    # get cluster logps
    view_state = X_L['view_state'][view_X]
    cluster_logps = numpy.array(su.determine_cluster_crp_logps(view_state))
    n_clusters = len(cluster_logps)

    # get X values
    x_values = M_c['column_metadata'][X]['code_to_value'].values()
    # get Y values
    y_values = M_c['column_metadata'][Y]['code_to_value'].values()

    # get components models for each cluster for columns X and Y
    component_models_X = [0]*n_clusters
    component_models_Y = [0]*n_clusters
    for i in range(n_clusters):
        cluster_models = su.create_cluster_model_from_X_L(M_c, X_L, view_X, i)
        component_models_X[i] = cluster_models[X]
        component_models_Y[i] = cluster_models[Y]

    def marginal_predictive_logps_by_cluster(value, component_models):
        return numpy.array([
            component_models[j].calc_element_predictive_logp(value)
            + cluster_logps[j]
            for j in range(n_clusters)])

    x_marginal_predictive_logps_by_cluster = \
        [marginal_predictive_logps_by_cluster(x, component_models_X)
         for x in x_values]

    # \sum_c P(x|c)P(c)
    x_net_marginal_predictive_logps = \
        [logsumexp(ps) for ps in x_marginal_predictive_logps_by_cluster]

    y_marginal_predictive_logps_by_cluster = \
        [marginal_predictive_logps_by_cluster(y, component_models_Y)
         for y in y_values]

    # \sum_c P(y|c)P(c)
    y_net_marginal_predictive_logps = \
        [logsumexp(ps) for ps in y_marginal_predictive_logps_by_cluster]

    MI = 0.0

    for (i,x) in enumerate(x_values):
        x_marginals = x_marginal_predictive_logps_by_cluster[i]
        for (j,y) in enumerate(y_values):
            y_marginals = y_marginal_predictive_logps_by_cluster[j]
            # cluster prob is double-counted in sum of marginals
            joint_predictive_logp_by_cluster = \
                x_marginals + y_marginals - cluster_logps

            # \sum_c P(x|c)P(y|c)P(c), Joint distribution
            joint_predictive_logp = logsumexp(joint_predictive_logp_by_cluster)

            MI += math.exp(joint_predictive_logp) * \
                  (joint_predictive_logp - \
                   (x_net_marginal_predictive_logps[i] + \
                    y_net_marginal_predictive_logps[j]))

    # ignore MI < 0
    if MI <= 0.0:
        MI = 0.0

    return MI



# estimates the mutual information for columns X and Y.
def estimate_MI_sample(X, Y, M_c, X_L, _X_D, get_next_seed, n_samples=1000):
    random_state = numpy.random.RandomState(get_next_seed())

    get_view_index = lambda which_column: X_L['column_partition']['assignments'][which_column]

    view_X = get_view_index(X)
    view_Y = get_view_index(Y)

    # independent
    if view_X != view_Y:
        return 0.0

    # get cluster logps
    view_state = X_L['view_state'][view_X]
    cluster_logps = su.determine_cluster_crp_logps(view_state)
    cluster_crps = numpy.exp(cluster_logps) # get exp'ed values for multinomial
    n_clusters = len(cluster_crps)

    # get components models for each cluster for columns X and Y
    component_models_X = [0]*n_clusters
    component_models_Y = [0]*n_clusters
    for i in range(n_clusters):
        cluster_models = su.create_cluster_model_from_X_L(M_c, X_L, view_X, i)
        component_models_X[i] = cluster_models[X]
        component_models_Y[i] = cluster_models[Y]

    # MI = 0.0    # mutual information
    MI = numpy.zeros(n_samples)
    weights = numpy.zeros(n_samples)

    for i in range(n_samples):
        # draw a cluster
        cluster_idx = numpy.nonzero(random_state.multinomial(1, cluster_crps))[0][0]

        # get a sample from each cluster
        x = component_models_X[cluster_idx].get_draw(get_next_seed())
        y = component_models_Y[cluster_idx].get_draw(get_next_seed())

        # calculate marginal logs
        Pxy = numpy.zeros(n_clusters)   # P(x,y), Joint distribution
        Px = numpy.zeros(n_clusters)    # P(x)
        Py = numpy.zeros(n_clusters)    # P(y)

        # get logp of x and y in each cluster. add cluster logp's
        for j in range(n_clusters):

            Px[j] = component_models_X[j].calc_element_predictive_logp(x)
            Py[j] = component_models_Y[j].calc_element_predictive_logp(y)
            Pxy[j] = Px[j] + Py[j] + cluster_logps[j]   # \sum_c P(x|c)P(y|c)P(c), Joint distribution
            Px[j] += cluster_logps[j]                   # \sum_c P(x|c)P(c)
            Py[j] += cluster_logps[j]                   # \sum_c P(y|c)P(c)

        # pdb.set_trace()

        # sum over clusters
        Px = logsumexp(Px)
        Py = logsumexp(Py)
        Pxy = logsumexp(Pxy)

        # add to MI
        # MI += Pxy - (Px + Py)
        MI[i] = Pxy - (Px + Py)
        weights[i] = Pxy

    # do weighted average with underflow protection
    # MI /= float(n_samples)
    Z = logsumexp(weights)
    weights = numpy.exp(weights-Z)
    MI_ret = numpy.sum(MI*weights)

    # ignore MI < 0
    if MI_ret <= 0.0:
        MI_ret = 0.0

    return MI_ret
