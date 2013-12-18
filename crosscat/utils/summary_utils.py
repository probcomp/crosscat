import numpy
#
import crosscat.utils.convergence_test_utils


def get_logscore(p_State):
    return p_State.get_marginal_logp()

def get_num_views(p_State):
    return len(p_State.get_X_D())

def get_column_crp_alpha(p_State):
    return p_State.get_column_crp_alpha()

def get_ari(p_State):
    # requires environment: {view_assignment_truth}
    # requires import: {crosscat.utils.convergence_test_utils}
    X_L = p_State.get_X_L()
    ctu = crosscat.utils.convergence_test_utils
    return ctu.get_column_ARI(X_L, view_assignment_truth)

def get_mean_test_ll(p_State):
    # requires environment {M_c, T, T_test}
    # requires import: {crosscat.utils.convergence_test_utils}
    X_L = p_State.get_X_L()
    X_D = p_State.get_X_D()
    ctu = crosscat.utils.convergence_test_utils
    return ctu.calc_mean_test_log_likelihood(M_c, T, X_L, X_D, T_test)

def get_column_partition_assignments(p_State):
    return p_State.get_X_L()['column_partition']['assignments']

def column_partition_assignments_to_f_z_statistic(column_partition_assignments):
    # FIXME: actually implement this
    get_num_views_over_iters = lambda vector: map(len, map(set, vector))
    intermediate = map(get_num_views_over_iters,
            column_partition_assignments.T)
    return numpy.array(intermediate).T

def default_reprocess_summaries_func(summaries_arr_dict):
    column_partition_assignments = summaries_arr_dict.pop('column_partition_assignments')
    f_z_statistic = column_partition_assignments_to_f_z_statistic(column_partition_assignments)
    summaries_arr_dict['f_z_statistic'] = f_z_statistic
    #
    return summaries_arr_dict
