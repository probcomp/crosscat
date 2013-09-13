#!/usr/bin/env bash


# settings
WORK_DIR=/tmp/
WHICH_CLUSTER=0.2.0


grep_hcluster=$(yolk | grep hcluster)
if [[ -z $grep_hcluster ]]; then
    pip install --download $WORK_DIR hcluster==$WHICH_CLUSTER
    cd $WORK_DIR
    tar xvfz hcluster-$WHICH_CLUSTER.tar.gz
    cd hcluster-$WHICH_CLUSTER
    perl -pi.bak -e "s/input\('Selection \[default=1\]:'\)/2/" setup.py
    python setup.py install
fi

