#!/usr/bin/python


import os

# old crosscat setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# venture setup.py
# from distutils.core import setup, Extension


def generate_sources(dir_files_tuples):
    sources = []
    for dir, files in dir_files_tuples:
        full_files = [
                os.path.join(dir, file)
                for file in files
                ]
        sources.extend(full_files)
    return sources


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
    ]
State_sources = generate_sources([
    (pyx_src_dir, State_pyx_sources),
    (cpp_src_dir, State_cpp_sources),
    ])


# create exts
ContinuousComponentModel_ext = Extension(
    "crosscat.ContinuousComponentModel",
    libraries = ['boost'],
    extra_compile_args = [],
    sources=ContinuousComponentModel_sources,
    include_dirs=include_dirs,
    language="c++")
MultinomialComponentModel_ext = Extension(
    "crosscat.MultinomialComponentModel",
    libraries = ['boost'],
    extra_compile_args = [],
    sources=MultinomialComponentModel_sources,
    include_dirs=include_dirs,
    language="c++")
State_ext = Extension(
    "crosscat.State",
    libraries = ['boost'],
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

packages = ['crosscat']
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
