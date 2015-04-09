#
#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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
import itertools
import inspect
from timeit import default_timer
import datetime
import random
import multiprocessing
import multiprocessing.pool
import threading

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
            print '%s took:\t% 7d ms' % (self.task, self.elapsed)

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

class int_generator(object):
    """Int generator with mutex."""
    def __init__(self, start=None):
        self.start = start
        if start is None:
            self.start = random.randrange(32767)
        self.next_i = self.start
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            self.next_i += 1
            return self.next_i

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
    print print_str

def ensure_listlike(input):
    if not isinstance(input, (list, tuple,)):
        input = [input]
    return input

def get_dict_as_text(parameters, join_with='\n'):
    create_line = lambda (key, value): key + ' = ' + str(value)
    lines = map(create_line, parameters.iteritems())
    text = join_with.join(lines)
    return text
