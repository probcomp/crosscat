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
from __future__ import print_function
import itertools
import inspect
from timeit import default_timer
import datetime
import random
import multiprocessing
import multiprocessing.pool
import numbers
import threading
import math
import six

#http://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class Timer(object):
    def __init__(self, task='action', verbose=True):
        self.task = task
        self.verbose = verbose
        self.timer = default_timer
        self.start = None
    def get_elapsed_secs(self):
        end = self.timer()
        return end - self.start
    def __enter__(self):
        self.start = self.timer()
        return self
    def __exit__(self, *args):
        self.elapsed_secs = self.get_elapsed_secs()
        self.elapsed = self.elapsed_secs * 1000 # millisecs
        if self.verbose:
            print('%s took:\t% 7d ms' % (self.task, self.elapsed))

class MapperContext(object):
    def __init__(self, do_multiprocessing=True, Pool=multiprocessing.Pool,
            *args, **kwargs):
        self.pool = None
        self.map = map
        if do_multiprocessing:
            self.pool = Pool(*args, **kwargs)
            self.map = self.pool.map
            pass
        return

    def __enter__(self):
        return self.map

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            pass
        return False

def int_generator(prngstate=None):
    if prngstate is None:
        prngstate = random.Random(32767)
    elif isinstance(prngstate, numbers.Integral):
        prngstate = random.Random(prngstate)
    else:
        if not hasattr(prngstate, 'randint'):
            raise TypeError('prngstate must be None, an integer, or a PRNG '
                            'with a randint method.  (E.g., random.Random '
                            'or numpy.random.RandomState instance.)')
    lock = threading.Lock()
    for _ in xrange(2**62):
        with lock:
            yield prngstate.randint(0, 2147483646)

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))

def divide_N_fairly(N, num_partitions):
    _n = N / num_partitions
    ns = [_n] * num_partitions
    delta = N - sum(ns)
    for idx in range(delta):
        ns[idx] += 1
    return ns

# introspection helpers
def is_obj_method_name(obj, method_name):
    attr = getattr(obj, method_name)
    is_method = inspect.ismethod(attr)
    return is_method
#
def get_method_names(obj):
    is_this_obj_method_name = lambda method_name: \
        is_obj_method_name(obj, method_name)
    #
    this_obj_attrs = dir(obj)
    this_obj_method_names = filter(is_this_obj_method_name, this_obj_attrs)
    return this_obj_method_names
#
def get_method_name_to_args(obj):
    method_names = get_method_names(obj)
    method_name_to_args = dict()
    for method_name in method_names:
        method = obj.__dict__[method_name]
        arg_str_list = inspect.getargspec(method).args[1:]
        method_name_to_args[method_name] = arg_str_list
    return method_name_to_args

def get_getname(name):
    return lambda in_dict: in_dict[name]

def print_ts(in_str):
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_str = '%s:: %s' % (now_str, in_str)
    print(print_str)

def ensure_listlike(input):
    if not isinstance(input, (list, tuple,)):
        input = [input]
    return input

def get_dict_as_text(parameters, join_with='\n'):
    create_line = lambda key_value: key_value[0] + ' = ' + str(key_value[1])
    lines = map(create_line, six.iteritems(parameters))
    text = join_with.join(lines)
    return text

def logsumexp(array):
    if len(array) == 0:
        return float('-inf')
    m = max(array)

    # m = +inf means addends are all +inf, hence so are sum and log.
    # m = -inf means addends are all zero, hence so is sum, and log is
    # -inf.  But if +inf and -inf are among the inputs, or if input is
    # NaN, let the usual computation yield a NaN.
    if math.isinf(m) and min(array) != -m and \
       all(not math.isnan(a) for a in array):
        return m

    # Since m = max{a_0, a_1, ...}, it follows that a <= m for all a,
    # so a - m <= 0; hence exp(a - m) is guaranteed not to overflow.
    return m + math.log(sum(math.exp(a - m) for a in array))

def logmeanexp(array):
    inf = float('inf')
    if len(array) == 0:
        # logsumexp will DTRT, but math.log(len(array)) will fail.
        return -inf

    # Treat -inf values as log 0 -- they contribute zero to the sum in
    # logsumexp, but one to the count.
    #
    # If we pass -inf values through to logsumexp, and there are also
    # +inf values, then we get NaN -- but if we had averaged exp(-inf)
    # = 0 and exp(+inf) = +inf, we would sensibly get +inf, whose log
    # is still +inf, not NaN.  So strip -inf values first.
    #
    # Can't say `a > -inf' because that excludes NaNs, but we want to
    # include them so they propagate.
    noninfs = [a for a in array if not a == -inf]

    # probs = map(exp, logprobs)
    # log(mean(probs)) = log(sum(probs) / len(probs))
    #   = log(sum(probs)) - log(len(probs))
    #   = log(sum(map(exp, logprobs))) - log(len(logprobs))
    #   = logsumexp(logprobs) - log(len(logprobs))
    return logsumexp(noninfs) - math.log(len(array))
