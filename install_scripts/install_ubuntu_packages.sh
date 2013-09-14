#!/usr/bin/env bash


# test for root
if [[ "$USER" != "root" ]]; then
	echo "$0 must be executed as root"
	exit;
fi


bash update_git.sh
bash install_cx_freeze.sh
bash install_boost.sh

pip install virtualenv virtualenvwrapper

apt-get build-dep -y python-numpy python-matplotlib python-scipy

