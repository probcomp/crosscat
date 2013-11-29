#!/usr/bin/python
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


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
include_dirs = ['cpp_code/include/CrossCat']


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
RoundedContinuousComponentModel_pyx_sources = ['RoundedContinuousComponentModel.pyx']
RoundedContinuousComponentModel_cpp_sources = [
        'utils.cpp',
        'numerics.cpp',
        'RandomNumberGenerator.cpp',
        'ComponentModel.cpp',
        'ContinuousComponentModel.cpp',
        'RoundedContinuousComponentModel.cpp',
        ]
RoundedContinuousComponentModel_sources = generate_sources([
    (pyx_src_dir, RoundedContinuousComponentModel_pyx_sources),
    (cpp_src_dir, RoundedContinuousComponentModel_cpp_sources),
    ])
#
#
DoublyBoundedRoundedContinuousComponentModel_pyx_sources = ['DoublyBoundedRoundedContinuousComponentModel.pyx']
DoublyBoundedRoundedContinuousComponentModel_cpp_sources = [
        'utils.cpp',
        'numerics.cpp',
        'RandomNumberGenerator.cpp', 'ComponentModel.cpp',
        'ContinuousComponentModel.cpp',
        'RoundedContinuousComponentModel.cpp',
        'DoublyBoundedRoundedContinuousComponentModel.cpp',
        ]
DoublyBoundedRoundedContinuousComponentModel_sources = generate_sources([
    (pyx_src_dir, DoublyBoundedRoundedContinuousComponentModel_pyx_sources),
    (cpp_src_dir, DoublyBoundedRoundedContinuousComponentModel_cpp_sources),
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
RoundedContinuousComponentModel_ext = Extension(
    "crosscat.cython_code.RoundedContinuousComponentModel",
    libraries = ['boost_random'],
    extra_compile_args = [],
    sources=RoundedContinuousComponentModel_sources,
    include_dirs=include_dirs,
    language="c++")
DoublyBoundedRoundedContinuousComponentModel_ext = Extension(
    "crosscat.cython_code.DoublyBoundedRoundedContinuousComponentModel",
    libraries = ['boost_random'],
    extra_compile_args = [],
    sources=DoublyBoundedRoundedContinuousComponentModel_sources,
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
        RoundedContinuousComponentModel_ext,
        DoublyBoundedRoundedContinuousComponentModel_ext,
        MultinomialComponentModel_ext,
        State_ext,
        ]

packages = ['crosscat', 'crosscat.utils', 'crosscat.convergence_analysis', 'crosscat.jsonrpc_http', 'crosscat.cython_code']
setup(
        name='CrossCat',
        version='0.1',
        author='MIT.PCP',
        url='TBA',
        long_description='TBA.',
        packages=packages,
        package_dir={'crosscat':'crosscat/'},
        ext_modules=ext_modules,
        cmdclass = {'build_ext': build_ext}
        )
