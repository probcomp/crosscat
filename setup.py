#!/usr/bin/python
import os
import sys
try:
    from setuptools import setup
    from setuptools.command.build_py import build_py
    from setuptools.command.sdist import sdist
except ImportError:
    from distutils.core import setup
    from distutils.command.build_py import build_py
    from distutils.command.sdist import sdist
from distutils.extension import Extension

def get_version():
    with open('VERSION', 'rb') as f:
        version = f.read().strip()

    # Append the Git commit id if this is a development version.
    if version.endswith('+'):
        import re
        import subprocess
        version = version[:-1]
        tag = 'v' + version
        desc = subprocess.check_output([
            'git', 'describe', '--dirty', '--long', '--match', tag,
        ])
        match = re.match(r'^v([^-]*)-([0-9]+)-(.*)$', desc)
        assert match is not None
        verpart, revpart, localpart = match.groups()
        assert verpart == version
        # Local part may be g0123abcd or g0123abcd-dirty.  Hyphens are
        # not kosher here, so replace by dots.
        localpart = localpart.replace('-', '.')
        full_version = '%s.post%s+%s' % (verpart, revpart, localpart)
    else:
        full_version = version

    # Strip the local part if there is one, to appease pkg_resources,
    # which handles only PEP 386, not PEP 440.
    if '+' in full_version:
        pkg_version = full_version[:full_version.find('+')]
    else:
        pkg_version = full_version

    # Sanity-check the result.  XXX Consider checking the full PEP 386
    # and PEP 440 regular expressions here?
    assert '-' not in full_version, '%r' % (full_version,)
    assert '-' not in pkg_version, '%r' % (pkg_version,)
    assert '+' not in pkg_version, '%r' % (pkg_version,)

    return pkg_version, full_version

def write_version_py(path):
    try:
        with open(path, 'rb') as f:
            version_old = f.read()
    except IOError:
        version_old = None
    version_new = '__version__ = %r\n' % (full_version,)
    if version_old != version_new:
        print 'writing %s' % (path,)
        with open(path, 'wb') as f:
            f.write(version_new)

pkg_version, full_version = get_version()

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
    # XXX Kludge to work around distutils bug that passes all
    # CPython's C compiler flags to the C++ compiler, even ones that
    # don't make sense like -Wstrict-prototypes.
    nocxxflags = [
        '-Wstrict-prototypes',
    ]
    for flag in nocxxflags:
        if flag in self.compiler_so:
            self.compiler_so.remove(flag)
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
    + numpy_includes \
    + []


# specify sources
ContinuousComponentModel_pyx_sources = ['ContinuousComponentModel.pyx']
ContinuousComponentModel_cpp_sources = [
    'ComponentModel.cpp',
    'ContinuousComponentModel.cpp',
    'RandomNumberGenerator.cpp',
    'numerics.cpp',
    'utils.cpp',
    'weakprng.cpp',
]
ContinuousComponentModel_sources = generate_sources([
    (pyx_src_dir, ContinuousComponentModel_pyx_sources),
    (cpp_src_dir, ContinuousComponentModel_cpp_sources),
])
#
MultinomialComponentModel_pyx_sources = ['MultinomialComponentModel.pyx']
MultinomialComponentModel_cpp_sources = [
    'ComponentModel.cpp',
    'MultinomialComponentModel.cpp',
    'RandomNumberGenerator.cpp',
    'numerics.cpp',
    'utils.cpp',
    'weakprng.cpp',
]
MultinomialComponentModel_sources = generate_sources([
    (pyx_src_dir, MultinomialComponentModel_pyx_sources),
    (cpp_src_dir, MultinomialComponentModel_cpp_sources),
])
#
CyclicComponentModel_pyx_sources = ['CyclicComponentModel.pyx']
CyclicComponentModel_cpp_sources = [
    'ComponentModel.cpp',
    'CyclicComponentModel.cpp',
    'RandomNumberGenerator.cpp',
    'numerics.cpp',
    'utils.cpp',
    'weakprng.cpp',
]
CyclicComponentModel_sources = generate_sources([
    (pyx_src_dir, CyclicComponentModel_pyx_sources),
    (cpp_src_dir, CyclicComponentModel_cpp_sources),
])
#
State_pyx_sources = ['State.pyx']
State_cpp_sources = [
    'Cluster.cpp',
    'ComponentModel.cpp',
    'ContinuousComponentModel.cpp',
    'CyclicComponentModel.cpp',
    'DateTime.cpp',
    'MultinomialComponentModel.cpp',
    'RandomNumberGenerator.cpp',
    'State.cpp',
    'View.cpp',
    'numerics.cpp',
    'utils.cpp',
    'weakprng.cpp',
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
    'crosscat.cython_code',
    'crosscat.tests',
    'crosscat.tests.component_model_extensions',
    'crosscat.tests.unit_tests',
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

class local_build_py(build_py):
    def run(self):
        write_version_py(version_py)
        build_py.run(self)

# Make sure the VERSION file in the sdist is exactly specified, even
# if it is a development version, so that we do not need to run git to
# discover it -- which won't work because there's no .git directory in
# the sdist.
class local_sdist(sdist):
    def make_release_tree(self, base_dir, files):
        import os
        sdist.make_release_tree(self, base_dir, files)
        version_file = os.path.join(base_dir, 'VERSION')
        print('updating %s' % (version_file,))
        # Write to temporary file first and rename over permanent not
        # just to avoid atomicity issues (not likely an issue since if
        # interrupted the whole sdist directory is only partially
        # written) but because the upstream sdist may have made a hard
        # link, so overwriting in place will edit the source tree.
        with open(version_file + '.tmp', 'wb') as f:
            f.write('%s\n' % (pkg_version,))
        os.rename(version_file + '.tmp', version_file)

# XXX These should be attributes of `setup', but helpful distutils
# doesn't pass them through when it doesn't know about them a priori.
version_py = 'src/version.py'

setup(
    name='crosscat',
    version=pkg_version,
    author='MIT Probabilistic Computing Project',
    author_email='bayesdb@mit.edu',
    license='Apache License, Version 2.0',
    description='A domain-general, Bayesian method for analyzing high-dimensional data tables',
    url='https://github.com/probcomp/crosscat',
    long_description=long_description,
    packages=packages,
    package_dir={
        'crosscat': 'src',
        'crosscat.cython_code': 'src/cython_code',
        'crosscat.tests': 'src/tests',
        'crosscat.tests.component_model_extensions':
            'src/tests/component_model_extensions',
        'crosscat.tests.unit_tests': 'src/tests/unit_tests',
        'crosscat.utils': 'src/utils',
    },
    tests_require=[
        'pytest',
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,
        'build_py': local_build_py,
        'sdist': local_sdist,
    },
    zip_safe=False,
)
