from crosscat.utils import sample_utils as su
from crosscat.utils import data_utils as du
import matplotlib.pyplot as plt
import numpy as np

from crosscat.cython_code import ContinuousComponentModel as ccm


def run(args):
    ''' Demo of inference confidence

    `args` is a dict with the following keys

    n_samples : int
        the number of samples for imputation
    n_steps : int
        the number of corsscat iterations to use for confidence
    n_modes : int
        the number of modes from which to draw the sample data
    mean_std : float (greater than 0)
        the standard deviation between the means of the sample modes
    std_std : float (greater than 0)
        the standard deviation between the standard deviation of the modes
    seed : int
        RNG seed. If seed < 0, system time is used.

    '''
    n_samples = args['n_samples']
    n_steps = args['n_steps']
    n_modes = args['n_modes']
    mean_std = args['mean_std']
    std_std = args['std_std']
    seed = args['seed']

    if seed < 0:
        import time
        seed = int(time.time())

    np.random.seed(seed)
    means = np.random.normal(0, mean_std, n_modes)
    stds = 1/np.random.gamma(1, 1/std_std, n_modes)
    stds = [1.0]*n_modes

    samples = np.zeros(n_samples)

    for i in range(n_samples):
        mode = np.random.randint(n_modes)
        samples[i] = np.random.normal(means[mode], stds[mode])

    imputed = np.median(samples)
    conf, X_L_list, X_D_list = su.continuous_imputation_confidence(
        samples, imputed, (), n_steps=n_steps, return_metadata=True)

    results = {
        'config': args,
        'conf': conf,
        'samples': samples,
        'X_L_list': X_L_list,
        'X_D_list': X_D_list,
    }

    return results


def plot(results, filename=None):
    n_samples = results['config']['n_samples']
    samples = sorted(results['samples'])
    conf = results['conf']
    X_L = results['X_L_list'][0]
    X_D = results['X_D_list'][0]

    hgrm, _ = np.histogram(X_D[0], len(set(X_D[0])))
    max_mass_mode = np.argmax(hgrm)
    suffstats = X_L['view_state'][0]['column_component_suffstats'][0][max_mass_mode]

    counts = suffstats['N']
    sum_x = suffstats['sum_x']
    sum_x_sq = suffstats['sum_x_squared']
    scale = counts/results['config']['n_samples']
    component_model = ccm.p_ContinuousComponentModel(
        X_L['column_hypers'][0], counts, sum_x, sum_x_sq)

    plt.figure(facecolor='white')

    ax = plt.subplot(1, 2, 1)
    ax.hist(samples, min(31, int(n_samples/10)), normed=True, label='Samples',
            ec='none', fc='gray')
    T = [[x] for x in samples]
    M_c = du.gen_M_c_from_T(T, cctypes=['continuous'])

    xvals = np.linspace(np.min(samples), np.max(samples), 300)
    Q = [(n_samples, 0, x) for i, x in enumerate(xvals)]
    p = [su.simple_predictive_probability(M_c, X_L, X_D, [], [q]) for q in Q]
    p = np.array(p)
    ax.plot(xvals, np.exp(p), c='#bbbbbb',
            label='Predicitive probability', lw=3)
    p = [component_model.calc_element_predictive_logp(x) for x in xvals]
    ax.plot(xvals, np.exp(p)*scale, c='#222222', label='Summary mode',
            lw=3)
    plt.xlabel('Samples')
    plt.legend(loc=0)

    ax = plt.subplot(1, 2, 2)
    ax.bar([0, 1], [conf, 1.0-conf], fc='#333333', ec='none')
    ax.set_ylim([0, 1])
    ax.set_xlim([-.25, 2])
    ax.set_xticks([.5, 1.5])
    plt.ylabel('Probability mass')
    ax.set_xticklabels(['Summary mode', 'All other modes'])

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


if __name__ == "__main__":
    args = {
        'n_samples': 200,
        'n_steps': 100,
        'n_modes': 8,
        'mean_std': 5,
        'std_std': 0.001,
        'seed': -1,
    }
    results = run(args)
    plot(results)
