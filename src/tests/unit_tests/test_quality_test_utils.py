import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext
import crosscat.tests.component_model_extensions.MultinomialComponentModel as mcmext

import crosscat.tests.quality_test_utils as qtu

import numpy

import unittest

def main():
    unittest.main()

class TestKLDivergence(unittest.TestCase):
	def setUp(self):
		# took these from a random run of test_mixture_inference_quality.py
		self.X_L_cont = {'column_partition': {'assignments': [0], 'counts': [1], 'hypers': {b'alpha': 1.0}}, 'column_hypers': [{b'mu': 2.0466076397206323, b's': 0.40834476150565313, b'r': 1.0, b'fixed': 0.0, b'nu': 398.1071705534969}], 'view_state': [{'column_component_suffstats': [[{b'sum_x': 8.205354246888788, b'sum_x_squared': 16.833421146868414, b'N': 4.0}, {b'sum_x': 85.104360592631, b'sum_x_squared': 172.49753798755182, b'N': 42.0}, {b'sum_x': 2.0499065159901946, b'sum_x_squared': 4.202116724299058, b'N': 1.0}, {b'sum_x': 35.07094431469411, b'sum_x_squared': 72.35772359127941, b'N': 17.0}, {b'sum_x': 2.036911322430709, b'sum_x_squared': 4.14900773544642, b'N': 1.0}, {b'sum_x': 8.093437784278695, b'sum_x_squared': 16.37599412889093, b'N': 4.0}, {b'sum_x': 2.0427083497348937, b'sum_x_squared': 4.172657402076653, b'N': 1.0}, {b'sum_x': 223.86208760230662, b'sum_x_squared': 455.7146707336368, b'N': 110.0}, {b'sum_x': 75.95148533155684, b'sum_x_squared': 155.93814739763025, b'N': 37.0}, {b'sum_x': 236.25363041362004, b'sum_x_squared': 481.2396036033091, b'N': 116.0}, {b'sum_x': 673.1481299770528, b'sum_x_squared': 1466.7369323204234, b'N': 309.0}, {b'sum_x': 679.817505232329, b'sum_x_squared': 1291.2720311258447, b'N': 358.0}]], 'row_partition_model': {'counts': [4, 42, 1, 17, 1, 4, 1, 110, 37, 116, 309, 358], 'hypers': {b'alpha': 1.9952623149688797}}, 'column_names': [0]}]}
		self.X_D_cont = [[9, 9, 7, 8, 10, 11, 11, 9, 7, 11, 10, 11, 1, 11, 11, 9, 0, 9, 8, 10, 9, 3, 11, 11, 9, 7, 10, 10, 11, 9, 11, 10, 11, 11, 10, 9, 10, 11, 1, 11, 11, 8, 11, 11, 11, 11, 10, 10, 11, 1, 10, 7, 10, 11, 1, 7, 10, 10, 11, 11, 1, 10, 10, 9, 10, 10, 9, 1, 11, 11, 10, 11, 8, 11, 9, 10, 9, 10, 11, 11, 11, 10, 3, 9, 11, 10, 11, 11, 10, 11, 11, 10, 9, 11, 11, 11, 11, 11, 10, 8, 8, 9, 10, 1, 10, 11, 10, 7, 11, 10, 11, 11, 10, 11, 10, 11, 10, 10, 10, 11, 11, 10, 10, 11, 11, 9, 10, 10, 8, 11, 11, 7, 11, 10, 10, 8, 11, 11, 7, 10, 1, 10, 10, 10, 7, 9, 11, 11, 11, 5, 7, 1, 11, 11, 11, 10, 11, 10, 11, 11, 10, 10, 3, 8, 10, 11, 7, 1, 7, 11, 11, 11, 7, 10, 11, 11, 11, 10, 11, 11, 11, 10, 10, 10, 5, 10, 7, 11, 10, 11, 11, 10, 9, 7, 9, 11, 9, 11, 10, 7, 11, 9, 10, 7, 10, 11, 10, 10, 3, 9, 11, 10, 3, 11, 10, 7, 10, 10, 3, 10, 11, 11, 10, 7, 10, 3, 11, 8, 11, 1, 10, 11, 10, 11, 10, 7, 10, 10, 10, 9, 10, 10, 7, 10, 11, 1, 10, 11, 11, 9, 10, 11, 9, 11, 10, 10, 10, 1, 9, 7, 11, 11, 11, 11, 11, 10, 9, 10, 11, 10, 11, 10, 10, 7, 10, 10, 11, 9, 10, 11, 9, 11, 6, 9, 3, 11, 10, 11, 7, 10, 10, 10, 10, 11, 9, 10, 9, 9, 10, 10, 8, 7, 9, 11, 1, 11, 7, 11, 10, 11, 1, 11, 7, 11, 10, 10, 10, 9, 11, 11, 9, 11, 11, 9, 11, 11, 11, 11, 11, 10, 11, 11, 5, 11, 9, 11, 11, 11, 11, 1, 10, 7, 10, 11, 11, 10, 11, 10, 10, 9, 11, 3, 9, 11, 11, 11, 8, 9, 11, 7, 11, 1, 11, 10, 9, 11, 9, 7, 11, 11, 10, 8, 10, 10, 9, 11, 11, 9, 7, 7, 11, 11, 10, 11, 8, 11, 9, 10, 10, 8, 9, 11, 1, 10, 10, 11, 10, 7, 10, 10, 10, 9, 11, 3, 7, 8, 7, 11, 10, 11, 11, 7, 10, 10, 11, 10, 8, 1, 10, 10, 1, 10, 10, 10, 11, 10, 10, 7, 11, 10, 9, 1, 1, 11, 9, 9, 1, 10, 9, 8, 10, 7, 11, 10, 10, 3, 1, 7, 10, 1, 0, 11, 7, 10, 9, 11, 11, 10, 11, 1, 11, 11, 11, 11, 10, 10, 11, 1, 10, 11, 9, 10, 10, 11, 7, 11, 11, 1, 7, 1, 8, 11, 7, 10, 11, 10, 10, 10, 11, 7, 9, 10, 7, 7, 7, 10, 11, 10, 11, 8, 9, 7, 10, 11, 10, 9, 11, 7, 11, 7, 9, 11, 11, 9, 7, 10, 1, 10, 2, 10, 9, 7, 1, 11, 11, 9, 10, 11, 10, 11, 9, 11, 11, 10, 10, 4, 8, 7, 9, 11, 11, 7, 7, 11, 10, 11, 10, 9, 8, 7, 10, 1, 10, 7, 10, 11, 11, 7, 10, 10, 11, 9, 7, 10, 8, 11, 7, 8, 9, 11, 9, 10, 11, 7, 10, 10, 10, 11, 11, 10, 10, 7, 10, 10, 11, 10, 7, 7, 8, 11, 11, 10, 11, 7, 9, 11, 9, 11, 11, 10, 10, 10, 3, 8, 10, 9, 11, 9, 11, 7, 11, 10, 7, 10, 1, 10, 1, 7, 7, 9, 11, 7, 11, 9, 10, 11, 11, 11, 11, 10, 9, 9, 11, 9, 11, 10, 11, 11, 3, 7, 11, 7, 11, 11, 10, 9, 11, 11, 8, 11, 10, 10, 9, 11, 11, 3, 10, 9, 7, 7, 9, 11, 9, 7, 10, 10, 9, 10, 11, 10, 10, 11, 11, 11, 10, 1, 10, 7, 11, 9, 11, 10, 11, 9, 10, 11, 11, 10, 1, 11, 11, 11, 11, 11, 7, 11, 10, 11, 9, 10, 7, 11, 11, 9, 10, 11, 11, 11, 7, 10, 10, 9, 11, 10, 11, 10, 10, 11, 11, 7, 11, 7, 11, 10, 8, 11, 7, 0, 10, 11, 11, 11, 9, 7, 7, 10, 7, 11, 9, 11, 11, 10, 8, 11, 10, 11, 10, 10, 11, 11, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 10, 10, 7, 8, 10, 11, 11, 7, 11, 11, 10, 10, 11, 11, 10, 10, 7, 10, 10, 7, 10, 10, 10, 11, 11, 9, 11, 10, 10, 11, 7, 11, 11, 11, 1, 10, 11, 10, 11, 10, 10, 7, 9, 9, 10, 11, 11, 9, 9, 5, 10, 11, 10, 8, 7, 11, 11, 10, 11, 8, 10, 10, 11, 11, 11, 10, 9, 7, 9, 10, 7, 10, 10, 10, 10, 11, 11, 9, 10, 10, 7, 11, 7, 11, 10, 10, 7, 11, 11, 3, 11, 9, 1, 10, 9, 1, 11, 10, 7, 11, 9, 1, 11, 9, 11, 9, 11, 10, 11, 7, 10, 9, 10, 10, 9, 7, 10, 9, 9, 11, 11, 9, 7, 1, 8, 11, 11, 10, 10, 11, 7, 7, 11, 10, 9, 10, 11, 1, 10, 8, 11, 10, 3, 11, 7, 10, 11, 11, 9, 11, 10, 10, 10, 11, 10, 11, 10, 8, 10, 8, 11, 11, 11, 10, 11, 11, 9, 7, 10, 11, 9, 11, 10, 11, 11, 7, 10, 11, 11, 7, 10, 11, 11, 10, 11, 11, 7, 10, 10, 11, 10, 11, 11, 11, 11, 11, 10, 7, 11, 7, 10, 10, 10, 11, 10, 9, 9, 10, 11, 11, 8, 10, 9, 11, 11, 10, 11, 10, 10, 10, 10, 10, 11, 10, 10, 0, 11, 10, 11, 11, 11, 11, 11, 10, 11, 10, 10, 10, 3, 10]]
		self.M_c_cont = {'idx_to_name': {0: '0'}, 'column_metadata': [{'code_to_value': {}, 'value_to_code': {}, 'modeltype': 'normal_inverse_gamma'}], 'name_to_idx': {'0': 0}}
		self.params_cont = [{'mu': 2.0355962328365633, 'rho': 993.706739450366}, {'mu': 1.8962271679941651, 'rho': 948.5904506452995}, {'mu': 2.1747970062713993, 'rho': 953.1358923503657}]
		self.weights_cont = [1.0/3.0]*3

		self.X_L_mult = {'column_partition': {'assignments': [0], 'counts': [1], 'hypers': {b'alpha': 1.0}}, 'column_hypers': [{b'dirichlet_alpha': 1.0, b'K': 5.0, b'fixed': 0.0}], 'view_state': [{'column_component_suffstats': [[{b'1': 214.0, b'0': 122.0, b'3': 3.0, b'2': 20.0, b'4': 123.0, b'N': 482.0}, {b'1': 8.0, b'0': 1.0, b'2': 2.0, b'4': 1.0, b'N': 12.0}, {b'1': 6.0, b'0': 7.0, b'3': 1.0, b'2': 13.0, b'4': 43.0, b'N': 70.0}, {b'1': 2.0, b'0': 104.0, b'3': 9.0, b'2': 24.0, b'4': 49.0, b'N': 188.0}, {b'1': 2.0, b'0': 24.0, b'3': 50.0, b'2': 28.0, b'4': 33.0, b'N': 137.0}, {b'1': 2.0, b'0': 26.0, b'3': 5.0, b'2': 5.0, b'4': 9.0, b'N': 47.0}, {b'0': 9.0, b'3': 5.0, b'2': 1.0, b'4': 1.0, b'N': 16.0}, {b'1': 1.0, b'0': 6.0, b'2': 8.0, b'N': 15.0}, {b'1': 2.0, b'0': 7.0, b'3': 7.0, b'2': 4.0, b'4': 3.0, b'N': 23.0}, {b'0': 1.0, b'N': 1.0}, {b'0': 1.0, b'2': 1.0, b'N': 2.0}, {b'1': 3.0, b'0': 2.0, b'3': 1.0, b'N': 6.0}, {b'2': 1.0, b'N': 1.0}]], 'row_partition_model': {'counts': [482, 12, 70, 188, 137, 47, 16, 15, 23, 1, 2, 6, 1], 'hypers': {b'alpha': 1.5848931924611134}}, 'column_names': [0]}]}
		self.X_D_mult = [[2, 3, 3, 5, 0, 0, 2, 3, 4, 0, 0, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0, 0, 4, 4, 8, 8, 3, 0, 4, 0, 0, 3, 3, 2, 3, 6, 7, 0, 4, 0, 3, 0, 4, 0, 0, 4, 4, 4, 0, 4, 5, 0, 6, 0, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0, 4, 4, 3, 0, 4, 0, 1, 3, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 4, 5, 0, 4, 3, 3, 3, 0, 5, 6, 0, 3, 0, 2, 0, 3, 3, 5, 8, 0, 4, 2, 0, 3, 0, 4, 3, 0, 1, 0, 3, 0, 3, 4, 4, 0, 2, 0, 3, 0, 3, 11, 0, 4, 3, 4, 5, 0, 0, 0, 0, 5, 0, 0, 3, 0, 4, 3, 0, 4, 4, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 4, 3, 0, 11, 3, 7, 0, 3, 0, 4, 1, 3, 8, 0, 4, 0, 3, 0, 3, 3, 0, 3, 2, 0, 0, 0, 0, 0, 4, 0, 8, 0, 0, 3, 3, 0, 3, 0, 4, 0, 0, 3, 4, 0, 0, 0, 0, 3, 0, 0, 0, 4, 2, 4, 0, 0, 0, 0, 0, 0, 3, 0, 8, 5, 3, 0, 0, 4, 0, 0, 5, 4, 11, 2, 0, 0, 3, 0, 0, 0, 0, 0, 3, 4, 8, 3, 0, 5, 5, 2, 2, 0, 4, 0, 3, 5, 4, 8, 6, 0, 0, 4, 0, 3, 2, 3, 0, 0, 0, 0, 7, 3, 3, 0, 4, 2, 0, 5, 0, 0, 2, 2, 0, 0, 0, 0, 3, 0, 1, 0, 0, 3, 3, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 4, 2, 2, 0, 4, 7, 1, 5, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 3, 0, 3, 5, 3, 4, 4, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 4, 3, 0, 5, 4, 0, 8, 12, 0, 0, 4, 0, 3, 2, 0, 2, 6, 4, 0, 0, 0, 4, 2, 0, 0, 2, 4, 4, 0, 0, 3, 0, 8, 0, 0, 0, 4, 0, 0, 4, 2, 0, 0, 3, 0, 4, 0, 8, 0, 2, 0, 0, 3, 0, 0, 0, 4, 3, 6, 0, 2, 6, 0, 3, 0, 0, 3, 0, 5, 3, 0, 0, 3, 3, 3, 0, 4, 4, 3, 0, 0, 2, 2, 2, 0, 8, 0, 4, 0, 5, 3, 0, 0, 4, 0, 0, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 3, 2, 3, 0, 3, 3, 2, 0, 0, 3, 0, 0, 4, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 7, 4, 0, 3, 0, 3, 11, 4, 5, 0, 0, 5, 4, 0, 4, 0, 3, 0, 4, 2, 8, 4, 4, 10, 0, 3, 0, 0, 0, 2, 0, 0, 8, 3, 3, 0, 0, 0, 0, 0, 2, 4, 6, 3, 0, 2, 4, 3, 0, 0, 3, 0, 0, 4, 1, 0, 0, 8, 4, 0, 0, 5, 3, 3, 4, 3, 0, 3, 0, 4, 0, 3, 0, 0, 4, 3, 3, 0, 4, 0, 4, 4, 3, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 3, 2, 0, 0, 4, 0, 2, 5, 0, 0, 4, 0, 7, 6, 0, 0, 0, 0, 3, 4, 3, 2, 0, 0, 4, 0, 3, 0, 0, 0, 3, 4, 2, 3, 0, 3, 0, 0, 0, 4, 3, 5, 0, 5, 5, 2, 0, 4, 4, 2, 4, 0, 5, 3, 6, 3, 5, 3, 0, 5, 3, 3, 0, 0, 0, 0, 2, 2, 2, 4, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 5, 3, 8, 2, 7, 0, 0, 3, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 5, 0, 0, 0, 3, 0, 8, 0, 11, 3, 4, 3, 4, 4, 0, 8, 3, 0, 4, 4, 0, 2, 0, 0, 6, 0, 4, 4, 3, 3, 4, 0, 0, 0, 3, 0, 0, 0, 1, 3, 0, 0, 3, 0, 4, 4, 3, 3, 5, 4, 4, 0, 11, 5, 5, 0, 3, 0, 2, 3, 0, 4, 3, 3, 3, 3, 0, 0, 10, 3, 4, 8, 3, 0, 0, 0, 5, 5, 0, 4, 3, 7, 0, 0, 0, 0, 3, 5, 4, 3, 0, 0, 4, 0, 8, 3, 3, 0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 3, 4, 0, 8, 3, 3, 0, 0, 3, 0, 4, 3, 0, 4, 3, 0, 3, 3, 0, 0, 0, 4, 3, 3, 0, 6, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 4, 0, 0, 0, 0, 2, 3, 5, 0, 1, 3, 4, 4, 2, 0, 3, 3, 0, 3, 0, 0, 0, 2, 0, 7, 0, 0, 3, 0, 0, 5, 0, 2, 3, 0, 3, 0, 0, 0, 0, 3, 7, 5, 0, 3, 2, 0, 0, 0, 0, 6, 2, 3, 8, 0, 2, 4, 0, 0, 4, 0, 4, 0, 5, 0, 2, 0, 3, 0, 0, 7, 3, 0, 0, 0, 8, 3, 2, 0, 3, 4, 3, 0, 4, 3, 0, 0, 2, 3, 0, 0, 3, 0, 0, 4, 0, 0, 6, 2, 7, 0, 4, 0, 0, 4, 2, 0, 0, 0, 4, 6, 4, 0, 3, 4, 3, 0, 9, 0, 2, 3, 0, 0, 6, 7, 0, 5, 2, 0, 4, 5, 0, 4, 0, 0, 3, 4, 3, 0, 1, 2, 0, 3, 0, 2, 3, 0, 4, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 0, 3, 4, 0, 0, 3, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]]
		self.M_c_mult = {'idx_to_name': {0: '0'}, 'column_metadata': [{'code_to_value': {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4}, 'value_to_code': {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}, 'modeltype': 'symmetric_dirichlet_discrete'}], 'name_to_idx': {'0': 0}}
		self.params_mult = [{'weights': [0.25, 0.15000000000000002, 0.15000000000000002, 0.05000000000000002, 0.39999999999999997]}, {'weights': [0.44999999999999996, 0.39999999999999997, 1.3877787807814457e-17, 1.3877787807814457e-17, 0.15000000000000002]}, {'weights': [0.2, 0.2, 0.2, 0.2, 0.2]}]
		self.weights_mult = [1.0/3.0]*3		

	def test_should_output_single_float_continuous(self):
		kl = qtu.KL_divergence(ccmext.p_ContinuousComponentModel, 
			self.params_cont, self.weights_cont, self.M_c_cont, 
			self.X_L_cont, self.X_D_cont, n_samples=1000)

		assert isinstance(kl, float)
		assert kl >= 0.0

	def test_should_output_single_float_multinomial(self):
		kl = qtu.KL_divergence(mcmext.p_MultinomialComponentModel, 
			self.params_mult, self.weights_mult, self.M_c_mult, 
			self.X_L_mult, self.X_D_mult)

		assert isinstance(kl, float)
		assert kl >= 0.0


class TestGetMixtureSupport(unittest.TestCase):
	def setUp(self):
		self.params_list_normal = [
			{'mu':0.0, 'rho': 2.0},
			{'mu':3.0, 'rho': 2.0},
			{'mu':-3.0, 'rho': 2.0}
		]

		self.params_list_multinomial = [
			{'weights': [0.5, 0.5, 0.0, 0.0, 0.0]},
			{'weights': [0.0, 0.5, 0.5, 0.0, 0.0]},
			{'weights': [0.0, 0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0]},
		]

	def test_continuous_support_should_return_proper_number_of_bins(self):
		X = qtu.get_mixture_support('continuous', 
			ccmext.p_ContinuousComponentModel,
			self.params_list_normal, nbins=500)

		assert len(X) == 500

		X = qtu.get_mixture_support('continuous', 
			ccmext.p_ContinuousComponentModel,
			self.params_list_normal, nbins=522)

		assert len(X) == 522

	def test_multinomial_support_should_return_proper_number_of_bins(self):
		# support should be range(len(weights))
		X = qtu.get_mixture_support('multinomial', 
			mcmext.p_MultinomialComponentModel,
			self.params_list_multinomial)

		assert len(X) == len(self.params_list_multinomial[0]['weights'])

class TestGetMixturePDF(unittest.TestCase):
	def setUp(self):
		self.X_normal = numpy.array([0, .1 , .2 , .4, -.1, -.2])
		self.X_multinomial = numpy.array(list(range(5)))

		self.params_list_normal = [
			{'mu':0.0, 'rho': 2.0},
			{'mu':3.0, 'rho': 2.0},
			{'mu':-3.0, 'rho': 2.0}
		]

		self.params_list_multinomial = [
			{'weights': [0.5, 0.5, 0.0, 0.0, 0.0]},
			{'weights': [0.0, 0.5, 0.5, 0.0, 0.0]},
			{'weights': [0.0, 0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0]},
		]

		self.component_weights = [1.0/3.0]*3

	def test_should_return_value_for_each_element_in_X_contiuous(self):
		X = qtu.get_mixture_pdf(self.X_normal,
			ccmext.p_ContinuousComponentModel,
			self.params_list_normal, 
			self.component_weights)

		assert len(X) == len(self.X_normal)

	def test_should_return_value_for_each_element_in_X_multinomial(self):
		X = qtu.get_mixture_pdf(self.X_multinomial,
			mcmext.p_MultinomialComponentModel,
			self.params_list_multinomial, 
			self.component_weights)

		assert len(X) == len(self.X_multinomial)

	def test_component_weights_that_do_not_sum_to_1_should_raise_exception(self):
		self.assertRaises(ValueError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel,
			self.params_list_normal, [.1]*3)

	def test_length_component_weights_should_match_length_params_list(self):
		self.assertRaises(ValueError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel,
			self.params_list_normal, [.5]*2)

	def test_params_list_not_list_should_raise_exception(self):
		self.assertRaises(TypeError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel, 
			dict(), self.component_weights)

		self.assertRaises(TypeError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel,
			1.0, self.component_weights)

	def test_component_weights_not_list_should_raise_exception(self):
		self.assertRaises(TypeError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel,
			self.params_list_normal, dict())

		self.assertRaises(TypeError, qtu.get_mixture_pdf,
			self.X_normal, ccmext.p_ContinuousComponentModel,
			self.params_list_normal, 1.0)

class TestBincount(unittest.TestCase):
	def test_X_not_list_should_raise_exception(self):
		X = dict()
		self.assertRaises(TypeError, qtu.bincount, X)

		X = 2
		self.assertRaises(TypeError, qtu.bincount, X)

	def test_X_not_vector_should_raise_exception(self):
		X = numpy.zeros((2,2))
		self.assertRaises(ValueError, qtu.bincount, X)

	def test_bins_not_list_should_raise_exception(self):
		X = list(range(10))
		bins = dict()
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

		bins = 12
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

		bins = numpy.zeros(10)
		self.assertRaises(TypeError, qtu.bincount, X, bins=bins)

	def test_behavior_X_list(self):
		X = [0, 1, 2, 3]
		counts = qtu.bincount(X)
		assert counts == [1, 1, 1, 1]

		X = [1, 2, 2, 4, 6]
		counts = qtu.bincount(X)
		assert counts == [1, 2, 0, 1, 0, 1]

		bins = list(range(7))
		counts = qtu.bincount(X,bins)
		assert counts == [0, 1, 2, 0, 1, 0, 1]

		bins = [1,2,4,6]
		counts = qtu.bincount(X,bins)
		assert counts == [1, 2, 1, 1]

	def test_behavior_X_array(self):
		X = numpy.array([0, 1, 2, 3])
		counts = qtu.bincount(X)
		assert counts == [1, 1, 1, 1]

		X = numpy.array([1, 2, 2, 4, 6])
		counts = qtu.bincount(X)
		assert counts == [1, 2, 0, 1, 0, 1]

		bins = list(range(7))
		counts = qtu.bincount(X,bins)
		assert counts == [0, 1, 2, 0, 1, 0, 1]

		bins = [1,2,4,6]
		counts = qtu.bincount(X,bins)
		assert counts == [1, 2, 1, 1]

if __name__ == '__main__':
    main()
