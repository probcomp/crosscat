#!/usr/bin/env bash


if [[ "$USER" == "root" ]]; then
	apt-get install -y mercurial
else
	grep_cx_freeze=$(yolk -l | grep cx_Freeze)
	if [[ -z $grep_cx_freeze ]]; then
		hg clone https://bitbucket.org/anthony_tuininga/cx_freeze
		cd cx_freeze && python setup.py install
	fi
fi
