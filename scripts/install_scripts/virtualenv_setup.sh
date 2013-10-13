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
	-d      HOME=$HOME
	-c      CACHED_PACKAGES_DIR=$CACHED_PACKAGES_DIR
EOF
exit
}


#Process the arguments
while getopts hd:c: opt
do
	case "$opt" in
		h) usage;;
		d) HOME=$OPTARG;;
		c) CACHED_PACKAGES_DIR=$OPTARG;;
	esac
done


my_abs_path=$(readlink -f "$0")
my_dirname=$(dirname $my_abs_path)
project_location=$(dirname $(cd $my_dirname && git rev-parse --git-dir))
project_name=$(basename "$project_location")
requirements_filename="${project_location}/requirements.txt"


# ensure virtualenvwrapper is loaded
bashrc_has_virtualenvwrapper=$(grep WORKON_HOME "${HOME}/.bashrc")
if [[ -z "$bashrc_has_virtualenvwrapper" ]]; then
    echo "Setting up virtualenv via ~/.bashrc"  
    WORKON_HOME="${HOME}/.virtualenvs"
    wrapper_script=/usr/local/bin/virtualenvwrapper.sh
    cat -- >> "${HOME}/.bashrc" <<EOF
export WORKON_HOME="$WORKON_HOME"
source "$wrapper_script"
EOF
    # source so we can work with virtualenv below
    source "${HOME}/.bashrc"
fi


# ensure virtualenv exists for $project_name
has_project=$(workon | grep "$project_name")
if [[ -z $has_project ]]; then
    mkvirtualenv $project_name
    cdvirtualenv
    echo "cd $project_location" >> bin/postactivate
    cat -- >> ~/.bashrc <<EOF
export PYTHONPATH=\$PYTHONPATH:$project_location
EOF
    # source so we can work with virtualenv below
    source "${HOME}/.bashrc"
fi


workon $project_name
cd "$my_dirname"
bash install_python_packages.sh >virtualenv_setup.out 2>virtualenv_setup.err
