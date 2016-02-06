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
from six.moves import range
import copy
from collections import Counter
import numpy
import itertools
import six
#
import crosscat.cython_code.ContinuousComponentModel as CCM
import crosscat.cython_code.MultinomialComponentModel as MCM
import crosscat.cython_code.CyclicComponentModel as CYCM
import crosscat.utils.general_utils as gu
import crosscat.utils.data_utils as du
from crosscat.utils.general_utils import logsumexp
from crosscat.utils.general_utils import logmeanexp

class Bunch(dict):
    def __getattr__(self, key):
        if self.has_key(key):
            return self.get(key, None)
        else:
            raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value

Constraints = Bunch

def predictive_probability(M_c, X_L, X_D, Y, Q):
    # Evaluates the joint logpdf of crosscat columns. This is acheived by
    # invoking column_value_probability on univariate columns with
    # cascading the constraints (the chain rule).

    # Q (query): list of three element tuples where each tuple, (r,c,x)
    #  contains a row r; column, c; value x. All rows must be the same.
    # Y (contraints), follows an identical format.

    # The current interface does not allow the query columns to have different
    # row numbers, so this function will ensure the same constraint, pending
    # a formalization of the semantic meaning of predictive_probability of
    # arbitrary patterns of cells.

    # Permitting queries involving multiple hypothetical rows would
    # amount to demanding computation of the full Crosscat posterior.
    # That makes it reasonable to retain a restriction to at most one
    # hypothetical row, forcing the user to fall back to insertion and
    # analysis for more complex inquiries.
    queries = dict()
    for (row, col, val) in Q:
        if row != Q[0][0]:
            raise ValueError('Cannot specify different query rows.')
        if (row, col) in queries:
            raise ValueError('Cannot specify duplicate query columns.')
        if len(M_c['column_metadata']) <= col:
            raise ValueError('Cannot specify hypothetical query column.')
        queries[(row, col)] = val
    # Ensure consistency for nodes in both query and constraints.
    # This behavior is correct, even for real-valued datatypes. Conditional
    # probability is itself a complex topic, but consider random
    # variable X continuous. Then the conditional density of X f(s|X=t) is
    # 1 if s==t and 0 otherwise. Note change of the dominating measure from
    # Lebesgue to counting. The argument is not rigorous but correct.
    ignore = set()
    constraints = set()
    for (row, col, val) in Y:
        if (row, col) in constraints:
            raise ValueError('Cannot specify duplicate constraint row, column.')
        if (row, col) in queries:
            if queries[(row, col)] == val:
                ignore.add(col)
            else:
                return float('-inf')
        constraints.add((row, col))
    Y_prime = list(Y)
    # Chain rule.
    prob = 0
    for query in Q:
        if query[1] in ignore:
            continue
        r = simple_predictive_probability(M_c, X_L, X_D, Y_prime, [query])
        prob += float(r)
        Y_prime.append(query)
    return prob


# Q is a list of three element tuples where each tuple, (r,c,x) contains a
# row, r; a column, c; and a value x. The contraints, Y follow an identical format.
# Returns a numpy array where each entry, A[i] is the probability for query i given
# the contraints in Y.
def simple_predictive_probability(M_c, X_L, X_D, Y, Q):
    num_rows = len(X_D[0])
    num_cols = len(M_c['column_metadata'])
    query_row = Q[0][0]
    query_columns = [query[1] for query in Q]
    elements = [query[2] for query in Q]
    # enforce query rows all same row
    assert all([query[0]==query_row for query in Q])
    # enforce query columns observed column
    assert all([query_column<num_cols for query_column in query_columns])
    is_observed_row = query_row < num_rows

    x = []

    if not is_observed_row:
        x = simple_predictive_probability_unobserved(
            M_c, X_L, X_D, Y, query_row, query_columns, elements)
    else:
        x = simple_predictive_probability_observed(
            M_c, X_L, X_D, Y, query_row, query_columns, elements)

    return x


def simple_predictive_probability_observed(M_c, X_L, X_D, Y, query_row,
                                      query_columns, elements):
    n_queries = len(query_columns)

    answer = numpy.zeros(n_queries)

    for n in range(n_queries):
        query_column = query_columns[n]
        x = elements[n]

        # get the view to which this column is assigned
        view_idx = X_L['column_partition']['assignments'][query_column]
        # get cluster
        cluster_idx = X_D[view_idx][query_row]
        # get the cluster model for this cluster
        cluster_model = create_cluster_model_from_X_L(M_c, X_L, view_idx, cluster_idx)
        # get the specific cluster model for this column
        component_model = cluster_model[query_column]
        # construct draw conataints
        draw_constraints = get_draw_constraints(X_L, X_D, Y, query_row, query_column)

        # return the PDF value (exp)
        p_x = component_model.calc_element_predictive_logp_constrained(x, draw_constraints)

        answer[n] = p_x

    return answer

def simple_predictive_probability_unobserved(M_c, X_L, X_D, Y, query_row, query_columns, elements):

    n_queries = len(query_columns)

    answer = numpy.zeros(n_queries)
    # answers = numpy.array([])

    for n in range(n_queries):
        query_column = query_columns[n]
        x = elements[n]

        # get the view to which this column is assigned
        view_idx = X_L['column_partition']['assignments'][query_column]
        # get the logps for all the clusters (plus a new one) in this view
        cluster_logps = determine_cluster_logps(M_c, X_L, X_D, Y, query_row, view_idx)

        answers_n = numpy.zeros(len(cluster_logps))

        # cluster_logps should logsumexp to log(1)
        assert numpy.abs(logsumexp(cluster_logps)) < .0000001

        # enumerate over the clusters
        for cluster_idx in range(len(cluster_logps)):

            # get the cluster model for this cluster
            cluster_model = create_cluster_model_from_X_L(M_c, X_L, view_idx, cluster_idx)
            # get the specific cluster model for this column
            component_model = cluster_model[query_column]
            # construct draw conataints
            draw_constraints = get_draw_constraints(X_L, X_D, Y, query_row, query_column)

            # return the PDF value (exp)
            p_x = component_model.calc_element_predictive_logp_constrained(x, draw_constraints)


            answers_n[cluster_idx] = p_x+cluster_logps[cluster_idx]

        answer[n] = logsumexp(answers_n)

    return answer

##############################################################################

def row_structural_typicality(X_L_list, X_D_list, row_id):
    """
    Returns how typical the row is (opposite of how anomalous the row is).
    """
    count = 0
    assert len(X_L_list) == len(X_D_list)
    for X_L, X_D in zip(X_L_list, X_D_list):
        for r in range(len(X_D[0])):
            for c in range(len(X_L['column_partition']['assignments'])):
                if X_D[X_L['column_partition']['assignments'][c]][r] == X_D[X_L['column_partition']['assignments'][c]][row_id]:
                    count += 1
    return float(count) / (len(X_D_list) * len(X_D[0]) * len(X_L_list[0]['column_partition']['assignments']))


def column_structural_typicality(X_L_list, col_id):
    """
    Returns how typical the column is (opposite of how anomalous the column is).
    """
    count = 0
    for X_L in X_L_list:
        for c in range(len(X_L['column_partition']['assignments'])):
            if X_L['column_partition']['assignments'][col_id] == X_L['column_partition']['assignments'][c]:
                count += 1
    return float(count) / (len(X_L_list) * len(X_L_list[0]['column_partition']['assignments']))

def simple_predictive_probability_multistate(M_c, X_L_list, X_D_list, Y, Q):
    """
    Returns the simple predictive probability, averaged over each sample.
    """
    logprobs = [float(simple_predictive_probability(M_c, X_L, X_D, Y, Q))
        for X_L, X_D in zip(X_L_list, X_D_list)]
    return logmeanexp(logprobs)

def predictive_probability_multistate(M_c, X_L_list, X_D_list, Y, Q):
    """
    Returns the predictive probability, averaged over each sample.
    """
    logprobs = [float(predictive_probability(M_c, X_L, X_D, Y, Q))
        for X_L, X_D in zip(X_L_list, X_D_list)]
    return logmeanexp(logprobs)


#############################################################################

def similarity(M_c, X_L_list, X_D_list, given_row_id, target_row_id, target_column=None):
    """
    Returns the similarity of the given row to the target row, averaged over
    all the column indexes given by col_idxs.
    Similarity is defined as the proportion of times that two cells are in the same
    view and category.
    """
    score = 0.0

    ## Set col_idxs: defaults to all columns.
    if target_column:
        if type(target_column) == str:
            col_idxs = [M_c['name_to_idx'][target_column]]
        elif type(target_column) == list:
            col_idxs = target_column
        else:
            col_idxs = [target_column]
    else:
        col_idxs = M_c['idx_to_name'].keys()
    col_idxs = [int(col_idx) for col_idx in col_idxs]

    ## Iterate over all latent states.
    for X_L, X_D in zip(X_L_list, X_D_list):
        for col_idx in col_idxs:
            view_idx = X_L['column_partition']['assignments'][col_idx]
            if X_D[view_idx][given_row_id] == X_D[view_idx][target_row_id]:
                score += 1.0
    return score / (len(X_L_list)*len(col_idxs))

################################################################################
################################################################################

def simple_predictive_sample(M_c, X_L, X_D, Y, Q, get_next_seed, n=1):
    num_rows = len(X_D[0])
    num_cols = len(M_c['column_metadata'])
    query_row = Q[0][0]
    query_columns = [query[1] for query in Q]
    # enforce query rows all same row
    assert all([query[0]==query_row for query in Q])
    # enforce query columns observed column
    assert all([query_column<num_cols for query_column in query_columns])
    is_observed_row = query_row < num_rows
    x = []
    if not is_observed_row:
        x = simple_predictive_sample_unobserved(
            M_c, X_L, X_D, Y, query_row, query_columns, get_next_seed, n)
    else:
        x = simple_predictive_sample_observed(
            M_c, X_L, X_D, Y, query_row, query_columns, get_next_seed, n)
    # # more modular logic
    # observed_view_cluster_tuples = ()
    # if is_observed_row:
    #     observed_view_cluster_tuples = get_view_cluster_tuple(
    #         M_c, X_L, X_D, query_row)
    #     observed_view_cluster_tuples = [observed_view_cluster_tuples] * n
    # else:
    #     view_cluster_logps = determine_view_cluster_logps(
    #         M_c, X_L, X_D, Y, query_row)
    #     observed_view_cluster_tuples = \
    #         sample_view_cluster_tuples_from_logp(view_cluster_logps, n)
    # x = draw_from_view_cluster_tuples(M_c, X_L, X_D, Y,
    #                                   observed_view_cluster_tuples)
    return x

def simple_predictive_sample_multistate(M_c, X_L_list, X_D_list, Y, Q,
                                        get_next_seed, n=1):
    num_states = len(X_L_list)
    assert num_states==len(X_D_list)
    n_from_each = n / num_states
    n_sampled = n % num_states
    random_state = numpy.random.RandomState(get_next_seed())
    which_sampled = random_state.permutation(range(num_states))[:n_sampled]
    which_sampled = set(which_sampled)
    x = []
    for state_idx, (X_L, X_D) in enumerate(zip(X_L_list, X_D_list)):
        this_n = n_from_each
        if state_idx in which_sampled:
            this_n += 1
        this_x = simple_predictive_sample(M_c, X_L, X_D, Y, Q,
                                          get_next_seed, this_n)
        x.extend(this_x)
    return x

def simple_predictive_sample_observed(M_c, X_L, X_D, Y, which_row,
                                      which_columns, get_next_seed, n=1):
    # Reject attempts to query columns on which we are conditioned for
    # this observed row.  This amounts to asking Crosscat to predict
    # what value a column C would hold in an observed row, if it had a
    # specified value -- which is trivially itself.
    #
    # It is not clear that this is the product of any sensible
    # computation (unlike in the unobserved case, where we are
    # fantasizing new rows and want to present the whole fantasy), so
    # we reject it rather than provide the sole obvious answer, until
    # such time as someone argues it is the product of a sensible
    # computation.
    constrained_columns = set(col for row, col, _val in Y if row == which_row)
    assert not set(c for c in which_columns if c in constrained_columns), \
        'Query for constrained column in observed row makes no sense!'
    #
    def view_for(column):
        return X_L['column_partition']['assignments'][column]
    def cluster_model_for(view):
        cluster = X_D[view][which_row]
        # pull the suffstats, hypers, and marignal logP's for clusters
        return create_cluster_model_from_X_L(M_c, X_L, view, cluster)
    def component_model_for(column):
        return cluster_model_for(view_for(column))[column]
    samples_list = []
    for _ in range(n):
        this_sample_draws = []
        for which_column in which_columns:
            component_model = component_model_for(which_column)
            draw_constraints = get_draw_constraints(X_L, X_D, Y,
                                                    which_row, which_column)
            SEED = get_next_seed()
            draw = component_model.get_draw_constrained(SEED,draw_constraints)
            this_sample_draws.append(draw)
        samples_list.append(this_sample_draws)
    return samples_list

def names_to_global_indices(column_names, M_c):
    name_to_idx = M_c['name_to_idx']
    first_key = six.next(six.iterkeys(name_to_idx))
    # FIXME: str(column_name) is hack
    if isinstance(first_key, (six.text_type, six.binary_type)):
       column_names = map(str, column_names)
    return [name_to_idx[column_name] for column_name in column_names]

def extract_view_column_info(M_c, X_L, view_idx):
    view_state_i = X_L['view_state'][view_idx]
    column_names = view_state_i['column_names']
    # view_state_i ordering should match global ordering
    column_component_suffstats = view_state_i['column_component_suffstats']
    global_column_indices = names_to_global_indices(column_names, M_c)
    column_metadata = numpy.array([
        M_c['column_metadata'][col_idx]
        for col_idx in global_column_indices
        ])
    column_hypers = numpy.array([
            X_L['column_hypers'][col_idx]
            for col_idx in global_column_indices
            ])
    zipped_column_info = zip(column_metadata, column_hypers,
                             column_component_suffstats)
    zipped_column_info = dict(zip(global_column_indices, zipped_column_info))
    row_partition_model = view_state_i['row_partition_model']
    return zipped_column_info, row_partition_model

def get_column_info_subset(zipped_column_info, column_indices):
    column_info_subset = dict()
    for column_index in column_indices:
        if column_index in zipped_column_info:
            column_info_subset[column_index] = \
                zipped_column_info[column_index]
    return column_info_subset

def create_component_model(column_metadata, column_hypers, suffstats):
    modeltype = column_metadata['modeltype']
    if modeltype == 'normal_inverse_gamma':
        component_model = CCM.p_ContinuousComponentModel(
            column_hypers,
            count=suffstats.get(b'N', 0),
            sum_x=suffstats.get(b'sum_x', None),
            sum_x_squared=suffstats.get(b'sum_x_squared', None))
    elif modeltype == 'symmetric_dirichlet_discrete':
        # TODO Can we change the suffstats data structure not to
        # include the total count in the dictionary of per-item
        # counts, please?
        suffstats = copy.copy(suffstats)
        count = suffstats.pop(b'N', 0)
        component_model = MCM.p_MultinomialComponentModel(
            column_hypers, count=count, counts=suffstats)
    elif modeltype == 'vonmises':
        component_model = CYCM.p_CyclicComponentModel(
            column_hypers,
            count=suffstats.get(b'N', 0),
            sum_sin_x=suffstats.get(b'sum_sin_x', None),
            sum_cos_x=suffstats.get(b'sum_cos_x', None))
    else:
        raise ValueError('unknown modeltype: %r' % (modeltype,))
    return component_model

def create_cluster_model(zipped_column_info, row_partition_model,
                         cluster_idx):
    cluster_component_models = dict()
    for global_column_idx in zipped_column_info:
        column_metadata, column_hypers, column_component_suffstats = \
            zipped_column_info[global_column_idx]
        cluster_component_suffstats = column_component_suffstats[cluster_idx]
        component_model = create_component_model(
            column_metadata, column_hypers, cluster_component_suffstats)
        cluster_component_models[global_column_idx] = component_model
    return cluster_component_models

def create_empty_cluster_model(zipped_column_info):
    cluster_component_models = dict()
    for global_column_idx in zipped_column_info:
        column_metadata, column_hypers, _column_component_suffstats = \
            zipped_column_info[global_column_idx]
        component_model = create_component_model(column_metadata,
                                                 column_hypers, {b'N': None})
        cluster_component_models[global_column_idx] = component_model
    return cluster_component_models

def create_cluster_models(M_c, X_L, view_idx, which_columns=None):
    zipped_column_info, row_partition_model = extract_view_column_info(
        M_c, X_L, view_idx)
    if which_columns is not None:
        zipped_column_info = get_column_info_subset(
            zipped_column_info, which_columns)
    num_clusters = len(row_partition_model['counts'])
    cluster_models = []
    for cluster_idx in range(num_clusters):
        cluster_model = create_cluster_model(
            zipped_column_info, row_partition_model, cluster_idx
            )
        cluster_models.append(cluster_model)
    empty_cluster_model = create_empty_cluster_model(zipped_column_info)
    cluster_models.append(empty_cluster_model)
    return cluster_models

def determine_cluster_data_logp(cluster_model, cluster_sampling_constraints,
                                X_D_i, cluster_idx):
    logp = 0
    for column_idx, column_constraint_dict \
            in six.iteritems(cluster_sampling_constraints):
        if column_idx in cluster_model:
            other_constraint_values = []
            for other_row, other_value in column_constraint_dict['others']:
                if X_D_i[other_row]==cluster_idx:
                    other_constraint_values.append(other_value)
            this_constraint_value = column_constraint_dict['this']
            component_model = cluster_model[column_idx]
            logp += component_model.calc_element_predictive_logp_constrained(
                this_constraint_value, other_constraint_values)
    return logp

def get_cluster_sampling_constraints(Y, query_row):
    constraint_dict = dict()
    if Y is not None:
        for constraint in Y:
            constraint_row, constraint_col, constraint_value = constraint
            is_same_row = constraint_row == query_row
            if is_same_row:
                constraint_dict[constraint_col] = dict(this=constraint_value)
                constraint_dict[constraint_col]['others'] = []
        for constraint in Y:
            constraint_row, constraint_col, constraint_value = constraint
            is_same_row = constraint_row == query_row
            is_same_col = constraint_col in constraint_dict
            if is_same_col and not is_same_row:
                other = (constraint_row, constraint_value)
                constraint_dict[constraint_col]['others'].append(other)
    return constraint_dict

def get_draw_constraints(X_L, X_D, Y, draw_row, draw_column):
    constraint_values = []
    if Y is not None:
        column_partition_assignments = X_L['column_partition']['assignments']
        view_idx = column_partition_assignments[draw_column]
        X_D_i = X_D[view_idx]
        try:
            draw_cluster = X_D_i[draw_row]
        except IndexError:
            draw_cluster = None
        for constraint in Y:
            constraint_row, constraint_col, constraint_value = constraint
            try:
                constraint_cluster = X_D_i[constraint_row]
            except IndexError:
                constraint_cluster = None
            if (constraint_col == draw_column) \
                    and (constraint_cluster == draw_cluster):
                constraint_values.append(constraint_value)
    return constraint_values

def determine_cluster_data_logps(M_c, X_L, X_D, Y, query_row, view_idx):
    logps = []
    cluster_sampling_constraints = \
        get_cluster_sampling_constraints(Y, query_row)
    relevant_constraint_columns = cluster_sampling_constraints.keys()
    cluster_models = create_cluster_models(M_c, X_L, view_idx,
                                           relevant_constraint_columns)
    X_D_i = X_D[view_idx]
    for cluster_idx, cluster_model in enumerate(cluster_models):
        logp = determine_cluster_data_logp(
            cluster_model, cluster_sampling_constraints, X_D_i, cluster_idx)
        logps.append(logp)
    return logps

def determine_cluster_crp_logps(view_state_i):
    counts = view_state_i['row_partition_model']['counts']
    alpha = view_state_i['row_partition_model']['hypers'].get(b'alpha')
    counts_appended = numpy.append(counts, alpha)
    sum_counts_appended = sum(counts_appended)
    logps = numpy.log(counts_appended / float(sum_counts_appended))
    return logps

def determine_cluster_logps(M_c, X_L, X_D, Y, query_row, view_idx):
    view_state_i = X_L['view_state'][view_idx]
    cluster_crp_logps = determine_cluster_crp_logps(view_state_i)
    cluster_crp_logps = numpy.array(cluster_crp_logps)
    cluster_data_logps = determine_cluster_data_logps(M_c, X_L, X_D, Y,
                                                      query_row, view_idx)
    cluster_data_logps = numpy.array(cluster_data_logps)
    # We need to compute the vector of probabilities log[P(Z=j|Y)] where `Z`
    # is the row cluster, `Y` are the constraints, and `j` iterates from 1 to
    # the number of clusters (plus 1 for a new cluster) in the row partition of
    # `view_idx`. Mathematically:
    # log{P(Z=j|Y)} = log{P(Z=j)P(Y|Z=j) / P(Y) }
    #               = log{P(Z=j)} + log{P(Y|Z=j)} - log{sum_k(P(Z=k)P(Y|Z=k))}
    #               = cluster_crp_logps + cluster_data_logps - BAZ
    # The final term BAZ is computed by:
    # log{sum_k(P(Z=k)P(Y|Z=k))}
    # = log{sum_k(exp(log{P(Z=k)}+log{P(Y|Z=k)}))
    # = logsumexp(cluster_crp_logps + cluster_data_logps)
    cluster_logps = cluster_crp_logps + cluster_data_logps - \
        logsumexp(cluster_crp_logps + cluster_data_logps)

    return cluster_logps

def sample_from_cluster(cluster_model, random_state):
    sample = []
    for column_index in sorted(cluster_model.keys()):
        component_model = cluster_model[column_index]
        seed_i = random_state.randint(32767) # sys.maxint)
        sample_i = component_model.get_draw(seed_i)
        sample.append(sample_i)
    return sample

# LRU replacement, size 1
__cache = (-1, None)
def _cache_for(M_c):
    global __cache
    (cur_id, cur_cache) = __cache
    if cur_id == id(M_c):
        return cur_cache
    else:
        ans = {}
        __cache = (id(M_c), ans)
        return ans

def create_cluster_model_from_X_L(M_c, X_L, view_idx, cluster_idx):
    cache = _cache_for(M_c)
    key = ('cluster_model', id(X_L), view_idx, cluster_idx)
    if key in cache:
        return cache[key]
    else:
        ans = do_create_cluster_model_from_X_L(M_c, X_L, view_idx, cluster_idx)
        cache[key] = ans
        return ans

def do_create_cluster_model_from_X_L(M_c, X_L, view_idx, cluster_idx):
    zipped_column_info, row_partition_model = extract_view_column_info(
        M_c, X_L, view_idx)
    num_clusters = len(row_partition_model['counts'])
    if cluster_idx==num_clusters:
        # drew a new cluster
        return create_empty_cluster_model(zipped_column_info)
    else:
        return create_cluster_model(
            zipped_column_info, row_partition_model, cluster_idx)

def simple_predictive_sample_unobserved(M_c, X_L, X_D, Y, query_row,
                                        query_columns, get_next_seed, n=1):
    num_views = len(X_D)
    random_state = numpy.random.RandomState(get_next_seed())
    #
    cluster_logps_list = []
    # for each view
    for view_idx in range(num_views):
        # get the logp of the cluster of query_row in this view
        cluster_logps = determine_cluster_logps(M_c, X_L, X_D, Y, query_row,
                                                view_idx)
        cluster_logps_list.append(cluster_logps)
    #
    query_row_constraints = dict() if Y is None else \
        dict((col, val) for row, col, val in Y if row == query_row)
    def view_for(column):
        return X_L['column_partition']['assignments'][column]
    def cluster_model_for(view, cluster):
        return create_cluster_model_from_X_L(M_c, X_L, view, cluster)
    samples_list = []
    for _ in range(n):
        view_cluster_draws = dict()
        for view_idx, cluster_logps in enumerate(cluster_logps_list):
            probs = numpy.exp(cluster_logps)
            probs /= sum(probs)
            draw = numpy.nonzero(random_state.multinomial(1, probs))[0][0]
            view_cluster_draws[view_idx] = draw
        #
        def component_model_for(column):
            view = view_for(column)
            return cluster_model_for(view, view_cluster_draws[view])[column]
        this_sample_draws = []
        for query_column in query_columns:
            # If the caller specified this column in the constraints,
            # give the specified value -- otherwise by sampling from
            # the column's component model, we might get any other
            # value that is possible in this cluster, so that
            #
            #   SIMULATE x GIVEN x = 0
            #
            # might give 1 instead, which makes no sense.
            if query_column in query_row_constraints:
                draw = query_row_constraints[query_column]
            else:
                component_model = component_model_for(query_column)
                draw_constraints = get_draw_constraints(X_L, X_D, Y,
                                                        query_row, query_column)
                SEED = get_next_seed()
                draw = component_model.get_draw_constrained(SEED,
                                                            draw_constraints)
            this_sample_draws.append(draw)
        samples_list.append(this_sample_draws)
    return samples_list


def multinomial_imputation_confidence(samples, imputed, column_hypers_i):
    max_count = sum(numpy.array(samples) == imputed)
    confidence = float(max_count) / len(samples)
    return confidence

def get_continuous_mass_within_delta(samples, center, delta):
    num_samples = len(samples)
    num_within_delta = sum(numpy.abs(samples - center) < delta)
    mass_fraction = float(num_within_delta) / num_samples
    return mass_fraction


def continuous_imputation_confidence(samples, imputed,
                                     column_component_suffstats_i,
                                     n_steps=100, n_chains=1,
                                     return_metadata=False):
    # XXX: the confidence in continuous imputation is "the probability that
    # there exists a unimodal summary" which is defined as the proportion of
    # probability mass in the largest mode of a DPMM inferred from the simulate
    # samples. We use crosscat on the samples for a given number of iterations,
    # then calculate the proportion of mass in the largest mode.
    #
    # NOTE: The definition of confidence and its implementation do not agree.
    # The probability of a unimodal summary is P(k=1|X), where k is the number
    # of components in some infinite mixture model. I would describe the
    # current implementation as "Is there a mode with sufficient enough mass
    # that we can ignore the other modes". If this second formulation is to be
    # used, it means that we need to not use the median of all the samples as
    # the imputed value, but the median of the samples of the summary mode,
    # because the summary (the imputed value) should come from the summary
    # mode.
    #
    # There are a lot of problems with this second formulation.
    # 0. SLOW. Like, for real.
    # 1. Non-deterministic. The answer will be different given the same
    #   samples.
    # 2. Inaccurate. Approximate inference about approximate inferences.
    #   In practice confidences on the sample samples could be significantly
    #   different because the Gibbs sampler that underlies crosscat is
    #   susceptible to getting stuck in local maximum. Of course, this could be
    #   mitigated to some extent by using more chains, but things are slow
    #   enough as it is.
    # 3. Confidence (interval) has a distinct meaning to the people who will
    #   be using this software. A unimodal summary does not necessarily mean
    #   that inferences are within an acceptable range. We are going to need to
    #   be loud about this. Maybe there should be a notion of tolerance?
    #
    # An alternative: mutual predictive coverage
    # ------------------------------------------
    # Divide the number of samples in the intersection of the 90% CI's of each
    # component model by the number of samples in the union of the 90% CI's of
    # each component model.

    from crosscat.cython_code import State

    # XXX: assumes samples somes in as a 1-D numpy.array or 1-D list
    num_samples = float(len(samples))
    T = [[x] for x in samples]

    # XXX: This is a higly problematic consequence of the current definition of
    # confidence. If the number of samples is 1, then the confidence is always
    # 1 because there will be exactly 1 mode in the DPMM (recall the DPMM can
    # have, at maximum, as many modes at data points). I figure if we're going
    # to give a bad answer, we shoud give it quickly.
    if num_samples == 1:
        return 1.0

    confs = []
    tlist = ['column_hyperparameters',
             'row_partition_hyperparameters',
             'row_partition_assignments']
    M_c = du.gen_M_c_from_T(T, cctypes=['continuous'])

    if return_metadata:
        X_L_list = []
        X_D_list = []

    for _ in range(n_chains):
        ccstate = State.p_State(M_c, T)
        ccstate.transition(which_transitions=tlist, n_steps=n_steps)

        X_D = ccstate.get_X_D()

        assignment = X_D[0]
        num_cats = max(assignment)+1
        props = numpy.histogram(assignment, num_cats)[0]/num_samples
        confs.append(max(props))

        if return_metadata:
            X_L_list.append(ccstate.get_X_L())
            X_D_list.append(X_D)

    conf = numpy.mean(confs)
    if return_metadata:
        return conf, X_L_list, X_D_list
    else:
        return conf


def continuous_imputation(samples, get_next_seed):
    imputed = numpy.median(samples)
    return imputed

def multinomial_imputation(samples, get_next_seed):
    counter = Counter(samples)
    max_tuple = counter.most_common(1)[0]
    max_count = max_tuple[1]
    counter_counter = Counter(counter.values())
    num_max_count = counter_counter[max_count]
    imputed = max_tuple[0]
    if num_max_count >= 1:
        # if there is a tie, draw randomly
        max_tuples = counter.most_common(num_max_count)
        values = [max_tuple[0] for max_tuple in max_tuples]
        random_state = numpy.random.RandomState(get_next_seed())
        draw = random_state.randint(len(values))
        imputed = values[draw]
    return imputed

# FIXME: ensure callers aren't passing continuous, multinomial
modeltype_to_imputation_function = {
    'normal_inverse_gamma': continuous_imputation,
    'symmetric_dirichlet_discrete': multinomial_imputation,
    }

modeltype_to_imputation_confidence_function = {
    'normal_inverse_gamma': continuous_imputation_confidence,
    'symmetric_dirichlet_discrete': multinomial_imputation_confidence,
    }

def impute(M_c, X_L, X_D, Y, Q, n, get_next_seed, return_samples=False):
    # FIXME: allow more than one cell to be imputed
    assert len(Q)==1
    #
    col_idx = Q[0][1]
    modeltype = M_c['column_metadata'][col_idx]['modeltype']
    assert modeltype in modeltype_to_imputation_function
    if get_is_multistate(X_L, X_D):
        samples = simple_predictive_sample_multistate(M_c, X_L, X_D, Y, Q,
                                           get_next_seed, n)
    else:
        samples = simple_predictive_sample(M_c, X_L, X_D, Y, Q,
                                           get_next_seed, n)
    samples = numpy.array(samples).T[0]
    imputation_function = modeltype_to_imputation_function[modeltype]
    e = imputation_function(samples, get_next_seed)
    if return_samples:
        return e, samples
    else:
        return e

def get_confidence_interval(imputed, samples, confidence=.5):
    deltas = numpy.array(samples) - imputed
    sorted_abs_delta = numpy.sort(numpy.abs(deltas))
    n_samples = len(samples)
    lower_index = int(numpy.floor(confidence * n_samples))
    lower_value = sorted_abs_delta[lower_index]
    upper_value = sorted_abs_delta[lower_index + 1]
    interval = numpy.mean([lower_value, upper_value])
    return interval

def get_column_std(column_component_suffstats_i):
    N = sum(map(gu.get_getname('N'), column_component_suffstats_i))
    sum_x = sum(map(gu.get_getname('sum_x'), column_component_suffstats_i))
    sum_x_squared = sum(map(gu.get_getname('sum_x_squared'), column_component_suffstats_i))
    #
    exp_x = sum_x / float(N)
    exp_x_squared = sum_x_squared / float(N)
    col_var = exp_x_squared - (exp_x ** 2)
    col_std = col_var ** .5
    return col_std

def get_column_component_suffstats_i(M_c, X_L, col_idx):
    column_name = M_c['idx_to_name'][str(col_idx)]
    view_idx = X_L['column_partition']['assignments'][col_idx]
    view_state_i = X_L['view_state'][view_idx]
    local_col_idx = view_state_i['column_names'].index(column_name)
    column_component_suffstats_i = \
        view_state_i['column_component_suffstats'][local_col_idx]
    return column_component_suffstats_i

def impute_and_confidence(M_c, X_L, X_D, Y, Q, n, get_next_seed):
    # FIXME: allow more than one cell to be imputed
    assert len(Q)==1
    col_idx = Q[0][1]
    modeltype = M_c['column_metadata'][col_idx]['modeltype']
    imputation_confidence_function = \
        modeltype_to_imputation_confidence_function[modeltype]
    #
    imputed, samples = impute(M_c, X_L, X_D, Y, Q, n, get_next_seed,
                        return_samples=True)
    if get_is_multistate(X_L, X_D):
        X_L = X_L[0]
        X_D = X_D[0]
    column_component_suffstats_i = \
        get_column_component_suffstats_i(M_c, X_L, col_idx)
    imputation_confidence = \
        imputation_confidence_function(samples, imputed,
                                       column_component_suffstats_i)
    return imputed, imputation_confidence

def determine_replicating_samples_params(X_L, X_D):
    view_assignments_array = X_L['column_partition']['assignments']
    view_assignments_array = numpy.array(view_assignments_array)
    views_replicating_samples = []
    for view_idx, view_zs in enumerate(X_D):
        is_this_view = view_assignments_array == view_idx
        this_view_columns = numpy.nonzero(is_this_view)[0]
        this_view_replicating_samples = []
        for cluster_idx, cluster_count in six.iteritems(Counter(view_zs)):
            view_zs_array = numpy.array(view_zs)
            first_row_idx = numpy.nonzero(view_zs_array==cluster_idx)[0][0]
            Y = None
            Q = [
                (int(first_row_idx), int(this_view_column))
                for this_view_column in this_view_columns
                ]
            n = cluster_count
            replicating_sample = dict(
                Y=Y,
                Q=Q,
                n=n,
                )
            this_view_replicating_samples.append(replicating_sample)
        views_replicating_samples.append(this_view_replicating_samples)
    return views_replicating_samples

def get_is_multistate(X_L, X_D):
    if isinstance(X_L, (list, tuple)):
        assert isinstance(X_D, (list, tuple))
        assert len(X_L) == len(X_D)
        return True
    else:
        return False

def ensure_multistate(X_L_list, X_D_list):
    was_multistate = get_is_multistate(X_L_list, X_D_list)
    if not was_multistate:
        X_L_list, X_D_list = [X_L_list], [X_D_list]
    # NOTE: We must do deepcopy or changes made to X_L (but, curiously,
    # not X_D)  hereafter shall be present in the original variables in
    # the original scope.
    # FIXME: The culprits are likely sparsify_X_L and desparsify_X_L in
    # State.pyx
    return copy.deepcopy(X_L_list), copy.deepcopy(X_D_list), was_multistate

# def determine_cluster_view_logps(M_c, X_L, X_D, Y):
#     get_which_view = lambda which_column: \
#         X_L['column_partition']['assignments'][which_column]
#     column_to_view = dict()
#     for which_column in which_columns:
#         column_to_view[which_column] = get_which_view(which_column)
#     num_views = len(X_D)
#     cluster_logps_list = []
#     for view_idx in range(num_views):
#         cluster_logps = determine_cluster_logps(M_c, X_L, X_D, Y, view_idx)
#         cluster_logps_list.append(cluster_logp)
#     return cluster_view_logps
