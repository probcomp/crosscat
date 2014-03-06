import argparse
from functools import partial
#
import pylab
pylab.ion()
pylab.show()
#
from crosscat.LocalEngine import LocalEngine
import crosscat.utils.data_utils as du
import crosscat.utils.timing_test_utils as ttu
import crosscat.utils.convergence_test_utils as ctu


parser = argparse.ArgumentParser()
parser.add_argument('--gen_seed', default=0, type=int)
parser.add_argument('--num_rows', default=100, type=int)
parser.add_argument('--num_cols', default=4, type=int)
parser.add_argument('--num_clusters', default=5, type=int)
parser.add_argument('--num_views', default=1, type=int)
parser.add_argument('--n_steps', default=10, type=int)
args = parser.parse_args()
#
gen_seed = args.gen_seed
num_rows = args.num_rows
num_cols = args.num_cols
num_clusters = args.num_clusters
num_views = args.num_views
n_steps = args.n_steps
#
n_test = num_rows / 10


# generate data
T, M_c, M_r, gen_X_L, gen_X_D = ttu.generate_clean_state(gen_seed, num_clusters,
        num_cols, num_rows, num_views)
T_test = ctu.create_test_set(M_c, T, gen_X_L, gen_X_D, n_test, seed_seed=0)
engine = LocalEngine()
X_L, X_D = engine.initialize(M_c, M_r, T)
gen_mtll = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T_test)
gen_preplexity = ctu.calc_mean_test_log_likelihood(M_c, T, gen_X_L, gen_X_D, T)


# run inference
calc_perplexity = lambda p_State: \
    ctu.calc_mean_test_log_likelihood(M_c, T, p_State.get_X_L(),
            p_State.get_X_D(), T)
calc_test_log_likelihood = lambda p_State: \
    ctu.calc_mean_test_log_likelihood(M_c, T, p_State.get_X_L(),
            p_State.get_X_D(), T_test)
diagnostic_func_dict = dict(
        perplexity=calc_perplexity,
        test_log_likelihood=calc_test_log_likelihood,
        )
X_L, X_D, diagnostics_dict = engine.analyze(M_c, T, X_L, X_D,
        do_diagnostics=diagnostic_func_dict, n_steps=n_steps)


# plot
pylab.plot(diagnostics_dict['test_log_likelihood'], 'g')
pylab.plot(diagnostics_dict['perplexity'], 'r')
pylab.axhline(gen_mtll, color='k')
pylab.axhline(gen_preplexity, color='b')
