#!/usr/bin/python
import os
import sys
try:
    from setuptools import setup
except ImportError:
    print 'FAILED: from setuptools import setup'
    print 'TRYING: from distutils.core import setup'
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#
import numpy


# http://stackoverflow.com/a/18992595
ON_LINUX = 'linux' in sys.platform
if ON_LINUX:
    has_ccache = os.system('which ccache') == 0
    if has_ccache:
        os.environ['CC'] = 'ccache gcc'

# http://stackoverflow.com/a/13176803
# monkey-patch for parallel compilation
import multiprocessing
import multiprocessing.pool
def parallelCCompile(self, sources, output_dir=None, macros=None,
        include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None,
        depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                    depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N_cores = multiprocessing.cpu_count()
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N_cores).imap(_single_compile,objects))
    return objects
#
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile

def generate_sources(dir_files_tuples):
    sources = []
    for dir, files in dir_files_tuples:
        full_files = [
                os.path.join(dir, file)
                for file in files
                ]
        sources.extend(full_files)
    return sources


# make sure cwd is correct
this_file = os.path.abspath(__file__)
this_dir = os.path.split(this_file)[0]
os.chdir(this_dir)


# locations
pyx_src_dir = 'crosscat/cython_code'
cpp_src_dir = 'cpp_code/src'
include_dirs = ['cpp_code/include/CrossCat', numpy.get_include()]


# specify sources
ContinuousComponentModel_pyx_sources = ['ContinuousComponentModel.pyx']
ContinuousComponentModel_cpp_sources = [
        'utils.cpp',
        'numerics.cpp',
        'RandomNumberGenerator.cpp',
        'ComponentModel.cpp',
        'ContinuousComponentModel.cpp',
        ]
ContinuousComponentModel_sources = generate_sources([
    (pyx_src_dir, ContinuousComponentModel_pyx_sources),
    (cpp_src_dir, ContinuousComponentModel_cpp_sources),
    ])
#
MultinomialComponentModel_pyx_sources = ['MultinomialComponentModel.pyx']
MultinomialComponentModel_cpp_sources = [
    'utils.cpp',
    'numerics.cpp',
    'RandomNumberGenerator.cpp',
    'ComponentModel.cpp',
    'MultinomialComponentModel.cpp',
    ]
MultinomialComponentModel_sources = generate_sources([
    (pyx_src_dir, MultinomialComponentModel_pyx_sources),
    (cpp_src_dir, MultinomialComponentModel_cpp_sources),
    ])
#
State_pyx_sources = ['State.pyx']
State_cpp_sources = [
    'utils.cpp',
    'numerics.cpp',
    'RandomNumberGenerator.cpp',
    'DateTime.cpp',
    'View.cpp',
    'Cluster.cpp',
    'ComponentModel.cpp',
    'MultinomialComponentModel.cpp',
    'ContinuousComponentModel.cpp',
    'State.cpp',
    ]
State_sources = generate_sources([
    (pyx_src_dir, State_pyx_sources),
    (cpp_src_dir, State_cpp_sources),
    ])


# create exts
ContinuousComponentModel_ext = Extension(
    "crosscat.cython_code.ContinuousComponentModel",
    libraries = ['boost_random'],
    extra_compile_args = [],
    sources=ContinuousComponentModel_sources,
    include_dirs=include_dirs,
    language="c++")
MultinomialComponentModel_ext = Extension(
    "crosscat.cython_code.MultinomialComponentModel",
    libraries = ['boost_random'],
    extra_compile_args = [],
    sources=MultinomialComponentModel_sources,
    include_dirs=include_dirs,
    language="c++")
State_ext = Extension(
    "crosscat.cython_code.State",
    libraries = ['boost_random'],
    extra_compile_args = [],
    sources=State_sources,
    include_dirs=include_dirs,
    language="c++")
#
ext_modules = [
        ContinuousComponentModel_ext,
        MultinomialComponentModel_ext,
        State_ext,
        ]

packages = ['crosscat', 'crosscat.utils', 'crosscat.convergence_analysis',
    'crosscat.jsonrpc_http', 'crosscat.cython_code', 'crosscat.tests',
    'crosscat.tests.quality_tests', 'crosscat.tests.component_model_extensions']
setup(
        name='CrossCat',
        version='0.1',
        author='MIT.PCP',
        url='TBA',
        long_description='TBA.',
        packages=packages,
        dependency_links=['https://github.com/mit-probabilistic-computing-project/experiment_runner/tarball/master#egg=experiment_runner-0.1.1'],
        install_requires=['experiment_runner==0.1.1'],
        package_dir={'crosscat':'crosscat/'},
        ext_modules=ext_modules,
        cmdclass = {'build_ext': build_ext}
        )
