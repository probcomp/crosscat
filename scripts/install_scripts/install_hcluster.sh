#!/usr/bin/env bash
set -e
set -u


CACHED_PACKAGES_DIR=/tmp/CachedPythonPackages
WHICH_HCLUSTER=0.2.0
hcluster_tgz=hcluster-$WHICH_HCLUSTER.tar.gz
MISSING=missing
grep_hcluster=$MISSING


[ grep_hcluster=$(pip freeze | grep hcluster) ] || true
if [[ $grep_hcluster = "$MISSING" ]]; then
    mkdir -p $CACHED_PACKAGES_DIR
    if [[ ! -f $CACHED_PACKAGES_DIR/$hcluster_tgz ]]; then
        pip install --no-install --download $CACHED_PACKAGES_DIR hcluster==$WHICH_HCLUSTER
    fi
    cd $CACHED_PACKAGES_DIR
    tar xvfz $hcluster_tgz
    cd hcluster-$WHICH_HCLUSTER
    perl -pi.bak -e "s/input\('Selection \[default=1\]:'\)/2/" setup.py
    python setup.py install
fi

