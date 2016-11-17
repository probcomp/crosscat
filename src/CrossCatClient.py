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

import os
import inspect


class CrossCatClient(object):
    """A client interface that gives a singue interface to the various engines.

    Depending on the client_type, dispatch to appropriate engine constructor.
    """

    def __init__(self, engine):
        """Initialize client with given engine. Not to be called directly!"""
        self.engine = engine

    def __getattribute__(self, name):
        engine = object.__getattribute__(self, 'engine')
        attr = None
        if hasattr(engine, name):
            attr = getattr(engine, name)
        else:
            attr = object.__getattribute__(self, name)
        return attr


# Maybe this should be in CrossCatClient.__init__
def get_CrossCatClient(client_type, **kwargs):
    """Helper which instantiates the appropriate Engine and returns a Client"""
    client = None

    if client_type == 'local':
        import crosscat.LocalEngine as LocalEngine
        le = LocalEngine.LocalEngine(**kwargs)
        client = CrossCatClient(le)

    elif client_type == 'multiprocessing':
        import crosscat.MultiprocessingEngine as MultiprocessingEngine
        me =  MultiprocessingEngine.MultiprocessingEngine(**kwargs)
        client = CrossCatClient(me)

    else:
        raise Exception('unknown client_type: %s' % client_type)

    return client
