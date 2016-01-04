#!/bin/sh

set -Ceu

: ${PYTHON:=python}
: ${PY_TEST:=py.test}
case $PY_TEST in */*);; *) PY_TEST=`which "$PY_TEST"`;; esac

if [ ! -x "${PY_TEST}" ]; then
    printf >&2 'unable to find pytest\n'
    exit 1
fi

root=`cd -- "$(dirname -- "$0")" && pwd`

build_log=/tmp/crosscat-build-$$.log
(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    rm -f $build_log
    echo "Building, with cython warnings going to $build_log ..."
    ./pythenv.sh "$PYTHON" setup.py build > $build_log 2>&1 || \
	(cat $build_log && exit 1)
    if [ $# -eq 0 ]; then
        ./pythenv.sh "$PYTHON" "$PY_TEST" \
            src/tests/unit_tests \
            # end of tests
    else
        ./pythenv.sh "$PYTHON" "$PY_TEST" "$@"
    fi
)

runtests_log=/tmp/crosscat-runtests-$$.log
rm -f $runtests_log
echo "Running make runtests in cpp_code/ with log at $runtests_log ..."
(cd cpp_code && make runtests > $runtests_log 2>&1 && \
     echo "Passed!" || cat $build_log $runtests_log)
