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

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    ./pythenv.sh "$PYTHON" setup.py build
    if [ $# -eq 0 ]; then
        ./pythenv.sh "$PYTHON" "$PY_TEST" \
            src/tests/unit_tests \
            # end of tests
    else
        ./pythenv.sh "$PYTHON" "$PY_TEST" "$@"
    fi
)

(cd cpp_code && make runtests)
