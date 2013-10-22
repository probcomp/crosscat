from crosscat.LocalEngine import LocalEngine
import crosscat.utils.data_utils as data_utils


data_filename = 'T.csv'
inference_seed = 0
num_full_transitions = 10

# read the data table into internal json representation
data_table, row_metadata, column_metadata, header = \
        data_utils.read_data_objects(data_filename)

# create an engine to run analysis, inference
engine = LocalEngine(seed=inference_seed)

# initialize markov chain samples
initial_latent_state, initial_latent_state_clustering = \
        engine.initialize(column_metadata, row_metadata, data_table)

# run markov chain transition kernels on samples
latent_state, latent_state_clustering = engine.analyze(column_metadata,
        data_table, initial_latent_state, initial_latent_state_clustering,
        n_steps=num_full_transitions)

