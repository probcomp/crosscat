#!/usr/bin/env bash
set -e


# test for root
if [[ "$USER" != "root" ]]; then
	echo "$0 must be executed as root"
	exit;
fi


function get_which_boost {
	this_release=$(lsb_release -r | awk '{print $NF}')
	if [[ "$this_release" > "12.04" ]]; then
		echo libboost-all-dev
	else
		echo libboost1.48-all-dev
	fi
}


which_boost=$(get_which_boost)


# update seems necessary, else get
# E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
apt-get update
# install system dependencies
# engine dependencies
apt-get build-dep -y python-numpy python-matplotlib python-scipy
apt-get remove -y python-numpy python-setuptools python-sphinx ipython
apt-get install -y $which_boost ccache
# doc dependencies
apt-get install -y doxygen
# python dependencies
wget https://bootstrap.pypa.io/get-pip.py -O- | python
pip install -U distribute
