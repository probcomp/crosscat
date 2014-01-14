#!/bin/bash


# Helper functions
function abort_on_error () {
	if [[ $? -ne "0" ]]; then
		echo FAILED: $1
		exit 1
	fi
}


# settings
repo_base=$(readlink -f $(dirname $(git rev-parse --git-dir)))
installed_cython_dir=/usr/local/lib/python2.7/dist-packages/crosscat/cython_code/


# build
cd $repo_base
python setup.py install
abort_on_error "Failed to build via setup.py"
echo "Build via setup.py passed"


# make sure shared libraries are linked
# FIXME; figure out why jenkins can't see shared libraries without this
cd $repo_base/crosscat/cython_code/
ln -s $installed_cython_dir/State.so || true
ln -s $installed_cython_dir/MultinomialComponentModel.so || true
ln -s $installed_cython_dir/ContinuousComponentModel.so || true


# run the tests
cd $repo_base/crosscat/tests/unit_tests
nosetests --with-xunit
exit


# Build and run tests. WORKSPACE is set by jenkins to /var/
# export PYTHONPATH=$PYTHONPATH:$WORKSPACE
# cd $WORKSPACE/$project_name
# make tests
# make cython
# cd $project_name/tests
# python /usr/bin/nosetests --with-xunit cpp_unit_tests.py cpp_long_tests.py test_sampler_enumeration.py
