#!/usr/bin/env bash
set -e


# test for root
if [[ "$USER" != "root" ]]; then
	echo "$0 must be executed as root"
	exit;
fi






# update seems necessary, else get
# E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
apt-get update
# install system dependencies
# engine dependencies
apt-get build-dep -y python-numpy python-matplotlib python-scipy
apt-get remove -y python-numpy python-setuptools python-sphinx ipython
apt-get install -y libboost1.48-all-dev ccache
# doc dependencies
apt-get install -y doxygen

# 
wget https://bootstrap.pypa.io/get-pip.py -O- | python
pip install -U distribute
