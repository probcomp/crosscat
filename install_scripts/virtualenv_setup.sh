

# print script usage
usage() {
    cat <<EOF
usage: $0 options
    
    VMware automation functions
    
    OPTIONS:
    -h      Show this message
    -u      USER=$USER
    -d      HOME=$HOME
    -p      CACHED_PACAKGES_DIR=$CACHED_PACKAGES_DIR
EOF
exit
}


#Process the arguments
while getopts hu:d:p: opt
do
    case "$opt" in
        h) usage;;
	u) USER=$OPTARG;;
    	d) HOME=$OPTARG;;
    	p) CACHED_PACAKGES_DIR=$OPTARG;;
    esac
done


# BEWARE: readlink doesn't work on macs
my_abs_path=$(readlink -f "$0")
project_location=$(readlink -f "$(dirname $my_abs_path)/..")
project_name=$(basename $project_location)


options=
if [[ ! -z $CACHED_PACKAGES_DIR ]]; then
    # argument must be the full path
    full_path=$(readlink -f $CACHED_PACKAGES_DIR)
    options=" --no-index --find-links file://${full_path}"
fi
echo "using options=$options"
sleep 5


# ensure virtualenvwrapper is available
bashrc_has_virtualenvwrapper=$(grep WORKON_HOME ${HOME}/.bashrc)
if [[ -z $bashrc_has_virtualenvwrapper ]]; then
    echo "Setting up virtualenv via ~/.bashrc"  
    WORKON_HOME=${HOME}/.virtualenvs
    wrapper_script=/usr/local/bin/virtualenvwrapper.sh
    # FIXME: unclear if this will work for jenkins
    cat -- >> ${HOME}/.bashrc <<EOF
export PYTHONPATH=\${PYTHONPATH}:${project_location}
export WORKON_HOME=$WORKON_HOME
source $wrapper_script
EOF
fi

# source what we wrote above so we can work with virtualenv below
source ${HOME}/.bashrc

# ensure ${project_name}/requirements.txt is satisfied for virtualenv 
has_project=$(workon | grep $project_name)
if [[ -z $has_project ]]; then
    mkvirtualenv $project_name
    cdvirtualenv
    echo "cd $project_location" >> bin/postactivate
    # make sure yolk exists
    pip install yolk
    # numpy is problematic: must exist beforehand
    WHICH_NUMPY=$(grep numpy== $project_location/requirements.txt | awk '{print $NF}')
    pip install $options -r <(echo $WHICH_NUMPY)
fi

# always install requirements.txt in case new dependencies have been added
workon $project_name
pip install $options -r requirements.txt

bash install_hcluster.sh

# consider always using CACHED_PACAKGES_DIR and defaulting to /tmp/
# how to set up CACHED_PACKAGES_DIR
# CACHED_PACKAGES_DIR=~/CachedPythonPackages
# mkdir -p $CACHED_PACKAGES_DIR
# pip install --download --no-install $CACHED_PACKAGES_DIR -r requirements.txt
# pip install --no-index --find-links=file://$CACHED_PACKAGES_DIR -r requirements.txt
