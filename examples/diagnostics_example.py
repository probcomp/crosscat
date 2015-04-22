# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
#
import numpy
import pylab
pylab.ion()
pylab.show()
#
import crosscat.LocalEngine as LE
import crosscat.MultiprocessingEngine as ME
import crosscat.IPClusterEngine as IPE
import crosscat.tests.plot_utils as pu
import crosscat.utils.data_utils as du
import crosscat.utils.convergence_test_utils as ctu
import crosscat.utils.timing_test_utils as ttu
import crosscat.utils.diagnostic_utils as su

# <codecell>

# settings
gen_seed = 0
inf_seed = 0
num_clusters = 4
num_cols = 32
num_views = 4
n_steps = 64
diagnostics_every_N= 2
n_test = 40
data_max_mean = 1
data_max_std = 1.
#
#num_rows = 800
#n_chains = 16
#config_filename = os.path.expanduser('~/.config/ipython/profile_ssh/security/ipcontroller-client.json')
#
num_rows = 100
n_chains = 2
config_filename = None


# generate some data
T, M_r, M_c, data_inverse_permutation_indices = du.gen_factorial_data_objects(
        gen_seed, num_clusters, num_cols, num_rows, num_views,
        max_mean=data_max_mean, max_std=data_max_std,
        send_data_inverse_permutation_indices=True)
view_assignment_truth, X_D_truth = ctu.truth_from_permute_indices(
        data_inverse_permutation_indices, num_rows, num_cols, num_views, num_clusters)
X_L_gen, X_D_gen = ttu.get_generative_clustering(M_c, M_r, T,
        data_inverse_permutation_indices, num_clusters, num_views)
T_test = ctu.create_test_set(M_c, T, X_L_gen, X_D_gen, n_test, seed_seed=0)
#
generative_mean_test_log_likelihood = ctu.calc_mean_test_log_likelihood(M_c, T,
        X_L_gen, X_D_gen, T_test)
ground_truth_lookup = dict(
        ARI=1.0,
        mean_test_ll=generative_mean_test_log_likelihood,
        num_views=num_views,
        )

# <codecell>

# create the engine
# engine = ME.MultiprocessingEngine(seed=inf_seed)
engine = IPE.IPClusterEngine(config_filename=config_filename, seed=inf_seed)

# <codecell>

# run inference
do_diagnostics = True
X_L_list, X_D_list = engine.initialize(M_c, M_r, T, n_chains=n_chains)
X_L_list, X_D_list, diagnostics_dict = engine.analyze(M_c, T, X_L_list, X_D_list,
        n_steps=n_steps, do_diagnostics=do_diagnostics,
        diagnostics_every_N=diagnostics_every_N,
        )

# <codecell>

# plot results
pu.plot_diagnostics(diagnostics_dict, hline_lookup=ground_truth_lookup)

# <codecell>

# demonstrate custom diagnostic functions
# each custom function must take only p_State as its argument
diagnostic_func_dict = dict(LE.default_diagnostic_func_dict)
def get_ari(p_State):
    # requires environment: {view_assignment_truth}
    # requires import: {crosscat.utils.convergence_test_utils}
    X_L = p_State.get_X_L()
    ctu = crosscat.utils.convergence_test_utils
    return ctu.get_column_ARI(X_L, view_assignment_truth)
# push the function and any arguments needed from the surrounding environment
args_dict = dict(
        get_ari=get_ari,
        view_assignment_truth=view_assignment_truth,
        )
engine.dview.push(args_dict, block=True)
diagnostic_func_dict['ARI'] = get_ari

# <codecell>

# run inference
do_diagnostics = diagnostic_func_dict
X_L_list, X_D_list = engine.initialize(M_c, M_r, T, n_chains=n_chains)
X_L_list, X_D_list, diagnostics_dict = engine.analyze(M_c, T, X_L_list, X_D_list,
        n_steps=n_steps, do_diagnostics=do_diagnostics,
        diagnostics_every_N=diagnostics_every_N,
        )

# <codecell>

# plot results
which_diagnostics = ['num_views', 'column_crp_alpha', 'ARI', 'f_z[0, 1]', 'f_z[0, D]']
pu.plot_diagnostics(diagnostics_dict, hline_lookup=ground_truth_lookup,
        which_diagnostics=which_diagnostics)
# pu.plot_views(numpy.array(T), X_D_gen, X_L_gen, M_c)

# <codecell>


