#!/usr/bin/env bash


# test for root
if [[ "$USER" != "root" ]]; then
	echo "$0 must be executed as root"
	exit;
fi


# for git with credentials caching (optional)
add-apt-repository -y ppa:git-core/ppa
apt-get update
apt-get install -y git
