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

import math
import pytest

import crosscat.utils.general_utils as gu

def relerr(expected, actual):
    return abs((actual - expected)/expected)

def test_logsumexp():
    inf = float('inf')
    nan = float('nan')
    with pytest.raises(OverflowError):
        math.log(sum(map(math.exp, range(1000))))
    assert relerr(999.4586751453871, gu.logsumexp(range(1000))) < 1e-15
    assert gu.logsumexp([]) == -inf
    assert gu.logsumexp([-1000.]) == -1000.
    assert gu.logsumexp([-1000., -1000.]) == -1000. + math.log(2.)
    assert relerr(math.log(2.), gu.logsumexp([0., 0.])) < 1e-15
    assert gu.logsumexp([-inf, 1]) == 1
    assert gu.logsumexp([-inf, -inf]) == -inf
    assert gu.logsumexp([+inf, +inf]) == +inf
    assert math.isnan(gu.logsumexp([-inf, +inf]))
    assert math.isnan(gu.logsumexp([nan, inf]))
    assert math.isnan(gu.logsumexp([nan, -3]))

def test_logmeanexp():
    inf = float('inf')
    nan = float('nan')
    assert gu.logmeanexp([]) == -inf
    assert relerr(992.550919866405, gu.logmeanexp(range(1000))) < 1e-15
    assert gu.logmeanexp([-1000., -1000.]) == -1000.
    assert relerr(math.log(0.5 * (1 + math.exp(-1.))),
            gu.logmeanexp([0., -1.])) \
        < 1e-15
    assert relerr(math.log(0.5), gu.logmeanexp([0., -1000.])) < 1e-15
    assert relerr(-3 - math.log(2.), gu.logmeanexp([-inf, -3])) < 1e-15
    assert relerr(-3 - math.log(2.), gu.logmeanexp([-3, -inf])) < 1e-15
    assert gu.logmeanexp([+inf, -3]) == +inf
    assert gu.logmeanexp([-3, +inf]) == +inf
    assert gu.logmeanexp([-inf, 0, +inf]) == +inf
    assert math.isnan(gu.logmeanexp([nan, inf]))
    assert math.isnan(gu.logmeanexp([nan, -3]))
    assert math.isnan(gu.logmeanexp([nan]))

def test_get_scc_from_tuples():
    constraints = [(1,2), (2,3)]
    assert gu.get_scc_from_tuples(constraints) == {
        1: (1, 2, 3),
        2: (1, 2, 3),
        3: (1, 2, 3),
    }

    constraints = [(1,2), (3,4), (5,7), (7,3)]
    assert gu.get_scc_from_tuples(constraints) == {
        1: (1, 2),
        2: (1, 2),
        3: (3, 4, 5, 7),
        4: (3, 4, 5, 7),
        5: (3, 4, 5, 7),
        7: (3, 4, 5, 7),
    }

    constraints = [(1,2), (3,4), (5,7), (7,3)]
    assert gu.get_scc_from_tuples(constraints) == {
        1: (1, 2),
        2: (1, 2),
        3: (3, 4, 5, 7),
        4: (3, 4, 5, 7),
        5: (3, 4, 5, 7),
        7: (3, 4, 5, 7),
    }

    constraints = [(1,1), (2,2), (3,3), (4,4)]
    assert gu.get_scc_from_tuples(constraints) == {
        1: (1,),
        2: (2,),
        3: (3,),
        4: (4,),
    }

    constraints = [(1,2), (3,4), (4,1)]
    assert gu.get_scc_from_tuples(constraints) == {
        1: (1,2,3,4),
        2: (1,2,3,4),
        3: (1,2,3,4),
        4: (1,2,3,4),
    }
