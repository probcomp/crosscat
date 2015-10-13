XXX THIS DOCUMENT IS OUT OF DATE AND NEEDS TO BE UPDATED XXX

- We are switching from unittest to pytest.
- Quality tests need to be more reliably automated.
- Anything that is not automatic should be made so.
- Anything that is a cruddy command-line tool should be made a library.

# Component model class extensions for quality testing

This document covers how to extend the cython class in python so that new data types can be added to quality tests.

## Adding the git hook

The hook script is `check.sh` in the crosscat root directory. To add a pre-commit hook, put

	#!/bin/sh
	set -Ceu
	exec ./check.sh

into `.git/hooks/pre-commit` and make it executable:

	chmod +x .git/hooks/pre-commit

The tests will run before each commit. You should see something like this:

	$ git commit -m "added tests and documentation"
	....................................................................
	----------------------------------------------------------------------
	Ran 68 tests in 0.054s

	OK
	[inference_testing_framework d551512] added tests and documentation
	 10 files changed, 705 insertions(+), 21 deletions(-)

## Quality testing considerations

### Unit tests

All unit tests use the `unittest` module and should go in the `unit_tests` folder. The git hook will automatically run all tests in the `unit_tests` folder.

**Note:** Quality tests are separate from unit_tests because they rely on random processes (and sometime fail for statistically valid reasons [mixing]) and take over a minute to run. They should not be included in the pre-commit hook.

### Error and goodness-of-fit

The quality tests depend on goodness-of-fit and error measures. The default error measure (between the original data and predictive samples) is the mean sum of squares, the default goodness-of-fit test is a 2-sample Kolmogorov–Smirnov test. These test are not appropriate for categorical (multinomial) data so a conditional statement adds Chi-square tests for discrete data. 

Other data types may require other test, in which case it might be a good idea to create a more robust conditional statement or to add specific error and goodness-of-fit utilities.

### Data generation

- The `gen_data` method in the `synthetic_data_generator` module may require additional arguments for your data type (see code for detailed documentation).

- The `gen_data` method depends on a separation coefficient, C, that determines how well-separated the component model distributions are. C=0 implies that the distributions are identical and C=1 implies that they are well separated. You will need to add a routine to `generate_separated_model_parameters` to accomplish this.

- The methods in `synthetic_data_generator` do a considerable amount of input validation. Please add to this when you add your data type. You will also need to add to the unit tests in `unit_tests/test_synthetic_data_generator.py`

## Extending component models for testing

This portion of the document will go over what methods and properties to add to your component model class.

- If they do not exist, you will need to add methods to the cython class that retrieve the sufficient statistics and hyperparameters of the component model. 

- For examples see `crosscat/tests/component_model_extensions`

## Added properties:

The following properties must be added

### `cctype`

The string value of the cctype, or variable type. For example, the current component models have cctypes `continuous` and `mutlinomial`. For consistency, this value should match the value used in BayesDB and in the state initialization code.

### `model_type`

The string value of the model type. For example, the current component models have model_type `normal_inverse_gammas` and `symmetric_dirichlet_discrete`. For consistency, this value should match the value used in the CrossCat metadata.

## Added Methods:

The following methods must be added

## constructors

###  from_parameters

	@classmethod
	def from_parameters(cls, N, data_params=default_data_parameters, hypers=None, gen_seed=0):

Initialize a continuous component model with sufficient statistics generated from random data.

**Inputs:**
- N: the number of data points
- data_params: (optional) a dict of distribution parameters
- hypers: (optional) a dict of hyperparameters
- gen_seed: (optional) an integer from which the rng is seeded


### from_data

	@classmethod
    def from_data(cls, X, hypers=None, gen_seed=0):

Initialize a continuous component model with sufficient statistics generated from data X

**Inputs:**
- X: a column of data (numpy)
- hypers: (optional) dict of hyperparameters
- gen_seed: (optional) a int to seed the rng

## Probability

### uncollapsed_likelihood

	uncollapsed_likelihood(self, X, parameters)

Calculates the score of the data X under this component model with given parameters. Likelihood * prior.

*FIXME: Maybe change the name to log_score?*

**Inputs:**
- X: A column of data (numpy)
- parameters: a dict of component model parameters

**Returns:**
- log_p: a float

### log_likelihood

	@staticmethod
	def log_likelihood(X, parameters):

Calculates the log likelihood of the data X given parameters

**Inputs:**
- X: a column of data (numpy)
- parameters: a dict of component model parameters

**Returns:**
- log_likelihood: float. the likelihood to the data X

### log_pdf

	@staticmethod
	def log_pdf(X, parameters):

 Calculates the pdf for each point in the data X given parameters

**Inputs:**
- X: a column of data (numpy)
- parameters: a dict of component model parameters

**Returns:**
- log_pdf: numpy.ndarray. the logpdf for each element in X

### cdf

cdf(X, parameters):

	@staticmethod
	def cdf(X, parameters):

Calculates the cdf for each point in the data X given parameters

**Inputs:**
- X: a column of data (numpy)
- parameters: a dict with the following keys

**Returns:**
- cdf: numpy.ndarray cdf of each element in X

## Sampling

### sample_parameters_given_hyper

	def sample_parameters_given_hyper(self, gen_seed=0):

Samples a set of component model parameter given the current hyperparameters

**Inputs:**
- gen_seed: integer used to seed the rng

**Returns:**
- params: dict of component model parameters

### draw_hyperparameters

	@staticmethod
    def draw_hyperparameters(X, n_draws=1, gen_seed=0):

Draws hyperparameters from the same distribution that generates the grid in the C++ code.

**Inputs:**
- X: a column of data (numpy)
- n_draws: (optional) the number of draws
- gen_seed: (optional) seed the rng

**Returns:**
- A list of dicts of draws where each entry has keys for each hyperparameter

### generate_data_from_parameters

	@staticmethod
    def generate_data_from_parameters(params, N, gen_seed=0):

Generates data from the distribution defined by params

**Inputs:**
- params: a dict of component model parameters
- N: number of data points
- gen_seed: (optional) integer seed for rng

**Returns:**
- X: numpy.ndarray of N data points

## Utility

### generate_discrete_support

**Continuous:**
	
	@staticmethod
    def generate_discrete_support(params, support=0.95, nbins=100):

**Discrete:**

	@staticmethod
    def generate_discrete_support(params):

If continuous, returns a nbins-length set of points along the support interval; if discrete and bounded, returns the entire support. Inputs will vary by data type, but the only required parameter should be the component model parameters; other parameters should have default values.

