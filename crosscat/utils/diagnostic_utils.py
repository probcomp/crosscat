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

def column_chain_to_ratio(column_chain_arr, j, i=0):
    chain_i_j = column_chain_arr[[i, j], :]
    is_same = numpy.diff(chain_i_j, axis=0)[0] == 0
    n_chains = len(is_same)
    is_same_count = sum(is_same)
    ratio = is_same_count / float(n_chains)
    return ratio

def column_partition_assignments_to_f_z_statistic(column_partition_assignments,
        j, i=0):
    iter_column_chain_arr = column_partition_assignments.transpose((1, 0, 2))
    helper = lambda column_chain_arr: column_chain_to_ratio(column_chain_arr, j, i)
    as_list = map(helper, iter_column_chain_arr)
    return numpy.array(as_list)[:, numpy.newaxis]

def default_reprocess_diagnostics_func(diagnostics_arr_dict):
    column_partition_assignments = diagnostics_arr_dict.pop('column_partition_assignments')
    # column_paritition_assignments are column, iter, chain
    D = column_partition_assignments.shape[0] - 1
    f_z_statistic_0_1 = column_partition_assignments_to_f_z_statistic(column_partition_assignments, 1, 0)
    f_z_statistic_0_D = column_partition_assignments_to_f_z_statistic(column_partition_assignments, D, 0)
    diagnostics_arr_dict['f_z[0, 1]'] = f_z_statistic_0_1
    diagnostics_arr_dict['f_z[0, D]'] = f_z_statistic_0_D
    #
    return diagnostics_arr_dict
