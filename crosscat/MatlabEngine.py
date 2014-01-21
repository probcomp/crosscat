#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
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

from numpy import *
from mlabwrap import mlab
import json
import random

import pdb

# NOTE: get mlabwrap from http://mlabwrap.sourceforge.net/

class MatlabEngine(object):
    """
    Interfaces the matlab engine with CrossCat
    Notes:
    1. The matlab logo may pop up as mlabwrap is initializing a session.
    2. X_D is always passed as an array, so there is no need to parse a string
    3. This module provides minimal support for the matlab code, just enough to 
    run the Geweke tests.
    """
    def __init__(self):
        # TODO: add matlab code directory to matlab path, or make user do it?
        try:
            ret = mlab.MLE_flag()
        except AttributeError:
            raise AttributeError("MatlabEngine: cannot find MATLAB code.")

    def initialize(self, M_c, M_r, T, initialization='from_the_prior',
            specified_s_grid=0, specified_mu_grid=0,
            row_initialization=-1, n_chains=1):


        if isinstance(specified_s_grid, list) or isinstance(specified_s_grid, tuple):
            # also convert for math change (Teh->Murphy)
            if len(specified_s_grid) == 0:
                specified_s_grid = array(specified_s_grid)/2.0
            else:
                specified_s_grid = 0
        
        if isinstance(specified_mu_grid,list) or isinstance(specified_mu_grid, tuple):
            if len(specified_mu_grid) == 0:
                specified_mu_grid = array(specified_mu_grid)
            else:
                specified_mu_grid = 0

        X_L_list = []
        X_D_list = []
        for i in range(n_chains):
            ret = mlab.MLE_initialize( array(T), array(specified_s_grid), array(specified_mu_grid))
            obj = json.loads(ret)

            X_L_list.append(obj['X_L'])
            X_D_list.append(obj['X_D'])

        if n_chains == 1:
            return X_L_list[0], X_D_list[0]
        else:
            return X_L_list, X_D_list
        

    def analyze(self, M_c, T, X_L, X_D, specified_s_grid=0, specified_mu_grid=0):

        is_multistate = isinstance(X_L, list)
        if is_multistate:
            num_states = len(X_L)
        else:
            num_states = 1
            X_L = [X_L]
            X_D = [X_D]


        if isinstance(specified_s_grid, list) or isinstance(specified_s_grid, tuple):
            # also convert for math change (Teh->Murphy)
            if len(specified_s_grid) == 0:
                specified_s_grid = array(specified_s_grid)/2.0
            else:
                specified_s_grid = 0
        
        if isinstance(specified_mu_grid,list) or isinstance(specified_mu_grid, tuple):
            if len(specified_mu_grid) == 0:
                specified_mu_grid = array(specified_mu_grid)
            else:
                specified_mu_grid = 0


        X_L_list = []
        X_D_list = []
        for i in range(num_states):
            # convert to X_L to string
            try:
                ret = mlab.MLE_analyze( array(T), json.dumps(X_L[i]), array(X_D[i]), array(specified_s_grid), array(specified_mu_grid))
            except:
                print "MLE_analyze: ml error"
                pdb.set_trace()

            obj = json.loads(ret)

            X_L_list.append( obj['X_L'] )
            X_D_list.append( obj['X_D'] )


        if not is_multistate:
            return X_L_list[0], X_D_list[0]
        else:
            return X_L_list, X_D_list

    def simple_predictive_sample(self, M_c, X_L, X_D, Y, Q, n=1):
        
        is_multistate = isinstance(X_L, list)
        if is_multistate:
            num_states = len(X_L)
            i = random.randrange(num_states)
        else:
            i = 0
            num_states = 1
            X_L = [X_L]
            X_D = [X_D]

        Q = array(Q)+1;

        try:
            S = mlab.MLE_predictive_sample(json.dumps(X_L[i]), array(X_D[i]), array(0), Q, array([n]))
        except:
            print 'spp_error'
            pdb.set_trace()

        return S.tolist()
