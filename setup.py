#!/usr/bin/python
import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension

def get_version():
    with open('VERSION', 'rb') as f:
        version = f.read().strip()

    # Append the Git commit id if this is a development version.
    if version.endswith('+'):
        tag = 'v' + version[:-1]
        try:
            import subprocess
            desc = subprocess.check_output([
                'git', 'describe', '--dirty', '--match', tag,
            ])
        except Exception:
            version += 'unknown'
        else:
            assert desc.startswith(tag)
            import re
            match = re.match(r'v([^-]*)-([0-9]+)-(.*)$', desc)
            if match is None:       # paranoia
                version += 'unknown'
            else:
                ver, rev, local = match.groups()
                version = '%s.post%s+%s' % (ver, rev, local.replace('-', '.'))
                assert '-' not in version

    return version

def write_version_py(path):
    try:
        with open(path, 'rb') as f:
            version_old = f.read()
    except IOError:
        version_old = None
    version_new = '__version__ = %r\n' % (version,)
    if version_old != version_new:
        try:
            with open(path, 'wb') as f:
                f.write(version_new)
        except IOError:
            pass
            # If this is a read-only filesystem, probably we're not making
            # changes we'll want to commit anyway.

version = get_version()
write_version_py("src/version.py")

try:
    from Cython.Distutils import build_ext
except ImportError:
    import distutils.command.build_ext
    class build_ext(distutils.command.build_ext.build_ext):
        def build_extension(self, extension):
            raise Exception('Cython is unavailable to compile .pyx files.')


try:
    import numpy
except ImportError:
    numpy_includes = []
else:
    numpy_includes = [numpy.get_include()]


boost_includes = []
if 'BOOST_ROOT' in os.environ:
    BOOST_ROOT = os.environ['BOOST_ROOT']
    boost_includes.append(os.path.join(BOOST_ROOT, 'include'))


# http://stackoverflow.com/a/18992595
ON_LINUX = 'linux' in sys.platform
if ON_LINUX:
    has_ccache = os.system('which ccache >/dev/null 2>/dev/null') == 0
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
pyx_src_dir = 'src/cython_code'
cpp_src_dir = 'cpp_code/src'
include_dirs = ['cpp_code/include/CrossCat'] \
    + boost_includes \
    + numpy_includes \
    + []


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
CyclicComponentModel_pyx_sources = ['CyclicComponentModel.pyx']
CyclicComponentModel_cpp_sources = [
    'utils.cpp',
    'numerics.cpp',
    'RandomNumberGenerator.cpp',
    'ComponentModel.cpp',
    'CyclicComponentModel.cpp',
]
CyclicComponentModel_sources = generate_sources([
    (pyx_src_dir, CyclicComponentModel_pyx_sources),
    (cpp_src_dir, CyclicComponentModel_cpp_sources),
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
    'CyclicComponentModel.cpp',
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
    'crosscat.cython_code.ContinuousComponentModel',
    extra_compile_args = [],
    sources=ContinuousComponentModel_sources,
    include_dirs=include_dirs,
    language='c++',
)
MultinomialComponentModel_ext = Extension(
    'crosscat.cython_code.MultinomialComponentModel',
    extra_compile_args = [],
    sources=MultinomialComponentModel_sources,
    include_dirs=include_dirs,
    language='c++',
)
CyclicComponentModel_ext = Extension(
    'crosscat.cython_code.CyclicComponentModel',
    extra_compile_args = [],
    sources=CyclicComponentModel_sources,
    include_dirs=include_dirs,
    language='c++',
)
State_ext = Extension(
    'crosscat.cython_code.State',
    extra_compile_args = [],
    sources=State_sources,
    include_dirs=include_dirs,
    language='c++',
)
#
ext_modules = [
    CyclicComponentModel_ext,
    ContinuousComponentModel_ext,
    MultinomialComponentModel_ext,
    State_ext,
]

packages = [
    'crosscat',
    'crosscat.utils',
    'crosscat.convergence_analysis',
    'crosscat.jsonrpc_http',
    'crosscat.cython_code',
    'crosscat.tests',
    'crosscat.tests.component_model_extensions',
]


# fall back long description if the README is missing
long_description = 'CrossCat is a domain-general, Bayesian method for analyzing ' +\
    'high-dimensional data tables. CrossCat estimates the full joint distribution ' +\
    'over the variables in the table from the data, via approximate inference in ' +\
    'a hierarchical, nonparametric Bayesian model, and provides efficient samplers ' +\
    'for every conditional distribution. CrossCat combines strengths of ' +\
    'nonparametric mixture modeling and Bayesian network structure learning: it ' +\
    'can model any joint distribution given enough data by positing latent ' +\
    'variables, but also discovers independencies between the observable variables.'

if os.path.exists('README.rst'):
    long_description = open('README.rst').read()

setup(
    name='crosscat',
    version=version,
    author='MIT Probabilistic Computing Project',
    author_email='bayesdb@mit.edu',
    license='Apache License, Version 2.0',
    description='A domain-general, Bayesian method for analyzing high-dimensional data tables',
    url='https://github.com/probcomp/crosscat',
    long_description=long_description,
    packages=packages,
    install_requires=[
        'cython>=0.20.1',
        'numpy>=1.7.0',
        'six',
    ],
    package_dir={
        'crosscat': 'src',
        'crosscat.binary_creation': 'src/binary_creation',
        'crosscat.cython_code': 'src/cython_code',
        'crosscat.jsonrpc_http': 'src/jsonrpc_http',
        'crosscat.tests': 'src/tests',
        'crosscat.tests.component_model_extensions':
            'src/tests/component_model_extensions',
        'crosscat.utils': 'src/utils',
    },
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,
    },
)
