#!/bin/bash


# set default values
CACHED_PACKAGES_DIR=/tmp/CachedPythonPackages


# print script usage
usage() {
	cat <<EOF
usage: $0 options	
	Set up postgres database
	OPTIONS:

	-h      Show this message
	-c      CACHED_PACKAGES_DIR=$CACHED_PACKAGES_DIR
EOF
exit
}


#Process the arguments
while getopts hd:c: opt
do
	case "$opt" in
		h) usage;;
		c) CACHED_PACKAGES_DIR=$OPTARG;;
	esac
done


my_abs_path=$(readlink -f "$0")
my_dirname=$(dirname $my_abs_path)
project_location=$(dirname $(cd $my_dirname && git rev-parse --git-dir))
project_name=$(basename "$project_location")
requirements_filename="${project_location}/requirements.txt"


options=
if [[ ! -z "$CACHED_PACKAGES_DIR" ]]; then
    full_path=$(readlink -f "$CACHED_PACKAGES_DIR")
    mkdir -p $full_path
    options=" --download-cache $full_path"
fi
echo "using options=$options"
sleep 5


pip_install() {
	which_requirement=$1
	if [[ -z $which_requirement ]]; then
		echo pip_install received no arguments!
		echo exiting script
	fi
	pip install $options -r <(grep ^$which_requirement $requirements_filename | awk '{print $NF}')
}

# install problematic packages
pip_install numpy
pip_install scipy
pip_install pandas
pip_install patsy

# always install requirements.txt in case new dependencies have been added
pip install $options -r $requirements_filename

cd "$my_dirname"
bash install_hcluster.sh
bash install_cx_freeze.sh
