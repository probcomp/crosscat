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
from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libcpp.map cimport map as cpp_map
from cython.operator import dereference


cdef extern from "string" namespace "std":
	cdef cppclass string:
		char* c_str()
		string(char*)

cdef cpp_string get_string(in_string):
	cdef cpp_string cps = string(in_string)
	return cps

cdef set_string_double_map(cpp_map[cpp_string, double] &out_map, in_map):
	for key in in_map:
		out_map[get_string(key)] = in_map[key]

cdef extern from "CyclicComponentModel.h":
	cdef cppclass CyclicComponentModel:
		cpp_map[cpp_string, double] _get_suffstats()
		cpp_map[cpp_string, double] get_hypers()
		int get_count()
		double score
		cpp_string to_string()
		double get_draw(int seed)
		double get_draw_constrained(int seed, vector[double] constraints)
		double get_predictive_pdf(double element, vector[double] constraints)
		double insert_element(double element)
		double remove_element(double element)
		double incorporate_hyper_update()
		double calc_marginal_logp()
		double calc_element_predictive_logp(double element)
		double calc_element_predictive_logp_constrained(double element, vector[double] constraints)
	CyclicComponentModel *new_CyclicComponentModel "new CyclicComponentModel" (cpp_map[cpp_string, double] &in_hypers)
	CyclicComponentModel *new_CyclicComponentModel "new CyclicComponentModel" (cpp_map[cpp_string, double] &in_hypers, int COUNT, double SUM_SIN_X, double SUM_COS_X)
	void del_CyclicComponentModel "delete" (CyclicComponentModel *ccm)

cdef class p_CyclicComponentModel:
	cdef CyclicComponentModel *thisptr
	cdef cpp_map[cpp_string, double] hypers
	def __cinit__(self, in_map, count=None, sum_sin_x=None, sum_cos_x=None):
		set_string_double_map(self.hypers, in_map)
		if count is None:
			self.thisptr = new_CyclicComponentModel(self.hypers)
		else:
			self.thisptr = new_CyclicComponentModel(self.hypers,
				count, sum_sin_x, sum_cos_x)
	def __dealloc__(self):
		del_CyclicComponentModel(self.thisptr)
	def get_draw(self, seed):
		return self.thisptr.get_draw(seed)
	def get_draw_constrained(self, seed, constraints):
		return self.thisptr.get_draw_constrained(seed, constraints)
	def get_hypers(self):
		return self.thisptr.get_hypers()
	def get_suffstats(self):
		return self.thisptr._get_suffstats()
	def get_count(self):
		return self.thisptr.get_count()
	def insert_element(self, element):
		return self.thisptr.insert_element(element)
	def remove_element(self, element):
		return self.thisptr.remove_element(element)
	def incorporate_hyper_update(self):
		return self.thisptr.incorporate_hyper_update()
	def calc_marginal_logp(self):
		return self.thisptr.calc_marginal_logp()
	def calc_element_predictive_logp(self, element):
		return self.thisptr.calc_element_predictive_logp(element)
	def calc_element_predictive_logp_constrained(self, element, constraints):
		return self.thisptr.calc_element_predictive_logp_constrained(element, constraints)
	def __repr__(self):
		return self.thisptr.to_string()
