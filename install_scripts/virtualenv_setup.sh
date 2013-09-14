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


options=
if [[ ! -z "$CACHED_PACKAGES_DIR" ]]; then
    full_path=$(readlink -f "$CACHED_PACKAGES_DIR")
    mkdir -p $full_path
    options=" --download-cache $full_path"
fi
echo "using options=$options"
sleep 5

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

>virtualenv_setup.out
>virtualenv_setup.err

# ensure virtualenv exists for $project_name
has_project=$(workon | grep "$project_name")
if [[ -z $has_project ]]; then
    mkvirtualenv $project_name
    cdvirtualenv
    echo "cd $project_location" >> bin/postactivate
    cat -- >> ~/.bashrc <<EOF
export PYTHONPATH=\$PYTHONPATH:$project_location
EOF
fi


workon $project_name

pip_install() {
	which_requirement=$1
	if [[ -z $which_requirement ]]; then
		echo pip_install received no arguments!
		echo exiting script
	fi
	pip install $options -r <(grep ^$which_requirement $requirements_filename | awk '{print $NF}') \
		>>virtualenv_setup.out 2>>virtualenv_setup.err
}

# install problematic packages
pip_install numpy
pip_install scipy
pip_install pandas
pip_install patsy

# always install requirements.txt in case new dependencies have been added
pip install $options -r requirements.txt \
	>>virtualenv_setup.out 2>>virtualenv_setup.err

bash install_hcluster.sh
bash -i install_cx_freeze.sh
