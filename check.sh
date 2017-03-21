#!/bin/sh

set -Ceu

: ${MAKE:=make}
: ${PYTHON:=python}

root=`cd -- "$(dirname -- "$0")" && pwd`

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    ./pythenv.sh "$PYTHON" setup.py build
    if [ $# -eq 0 ]; then
        ./pythenv.sh "$PYTHON" -m pytest --pyargs crosscat
    else
        ./pythenv.sh "$PYTHON" -m pytest "$@"
    fi
)

if cd cpp_code && "${MAKE}" runtests; then
    echo 'Passed!'
else
    status=$?
    echo 'Failed!'
    exit $status
fi
