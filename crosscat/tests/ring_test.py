import numpy
import pylab 
import random
import math

import pdb

import crosscat.utils.sample_utils as su
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State
import crosscat.LocalEngine as LE

def ring(n=200, width=.1):

    X = numpy.zeros((n,2))
    for i in range(n):
        angle = random.uniform(0.0,2.0*math.pi)
        distance = random.uniform(1.0-width,1.0)
        X[i,0] = math.cos(angle)*distance
        X[i,1] = math.sin(angle)*distance
    return X

def sample_from_view(M_c, X_L, X_D, get_next_seed):
    
    view_col = X_L['column_partition']['assignments'][0]
    view_col2 = X_L['column_partition']['assignments'][1]

    same_view = (view_col2 == view_col)

    view_state = X_L['view_state'][view_col]
    view_state2 = X_L['view_state'][view_col2]

    cluster_crps = numpy.exp(su.determine_cluster_crp_logps(view_state))
    cluster_crps2 = numpy.exp(su.determine_cluster_crp_logps(view_state2))

    assert( math.fabs(numpy.sum(cluster_crps) - 1) < .00000001 )

    cluster_idx1 = numpy.nonzero(numpy.random.multinomial(1, cluster_crps))[0][0]
    cluster_model1 = su.create_cluster_model_from_X_L(M_c, X_L, view_col, cluster_idx1)

    if same_view:
        cluster_idx2 = cluster_idx1
        cluster_model2 = cluster_model1
    else:
        cluster_idx2 = numpy.nonzero(numpy.random.multinomial(1, cluster_crps2))[0][0]
        cluster_model2 = su.create_cluster_model_from_X_L(M_c, X_L, view_col2, cluster_idx2)

    component_model1 = cluster_model1[0]
    x = component_model1.get_draw(get_next_seed())

    component_model2 = cluster_model2[1]
    y = component_model2.get_draw(get_next_seed())
        
    return x, y

# hack-a-sampler
def sample_data_from_crosscat(M_c, X_Ls, X_Ds, get_next_seed, n):

    X = numpy.zeros((n,2))
    n_samples = len(X_Ls)
    
    for i in range(n):
        cc = random.randrange(n_samples)
        x, y = sample_from_view(M_c, X_Ls[cc], X_Ds[cc], get_next_seed)
        
        X[i,0] = x
        X[i,1] = y

    return X

get_next_seed = lambda : random.randrange(200000)

N = 1000
n_chains = 4
n_steps = 400

T = ring(N)
T = T.tolist()
M_c = du.gen_M_c_from_T(T)
M_r = du.gen_M_r_from_T(T)

# sample from 2-column State
X_L_list = []
X_D_list = []
for _ in range(n_chains):
    state = State.p_State(M_c, T)
    state.transition(n_steps=n_steps)
    X_D_list.append(state.get_X_D())
    X_L_list.append(state.get_X_L())

T_h = sample_data_from_crosscat(M_c, X_L_list, X_D_list, get_next_seed, N)

# sample from engine with predictive_sample
engine = LE.LocalEngine()
X_L_list, X_D_list = engine.initialize(M_c, M_r, T, n_chains=n_chains);
X_L_list, X_D_list = engine.analyze(M_c, T, X_L_list, X_D_list, n_steps=n_steps)

Y = []
Q = [(N,0),(N,1)]
S = engine.simple_predictive_sample(M_c, X_L_list, X_D_list, Y, Q, n=N)

T_o = numpy.array(T)
T_i = numpy.array(S)

ax = pylab.subplot(1,3,1)
pylab.scatter( T_o[:,0], T_o[:,1], color='blue', edgecolor='none' )
pylab.ylabel("X")
pylab.ylabel("Y")
pylab.title("original")

pylab.subplot(1,3,2)
pylab.scatter( T_i[:,0], T_i[:,1], color='red', edgecolor='none' )
pylab.ylabel("X")
pylab.ylabel("Y")
pylab.xlim(ax.get_xlim())
pylab.ylim(ax.get_ylim())
pylab.title("simulated (LocalEngine)")

pylab.subplot(1,3,3)
pylab.scatter( T_h[:,0], T_h[:,1], color='red', edgecolor='none' )
pylab.ylabel("X")
pylab.ylabel("Y")
pylab.xlim(ax.get_xlim())
pylab.ylim(ax.get_ylim())
pylab.title("simulated (Hacked)")

pylab.savefig('predictive_sample_ring.png')