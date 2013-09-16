#!/usr/bin/env bash


if [[ "$USER" == "root" ]]; then
	apt-get install -y mercurial
else
	hg clone https://bitbucket.org/anthony_tuininga/cx_freeze
	cd cx_freeze && python setup.py install
fi
