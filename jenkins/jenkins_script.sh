#!/bin/bash

# default values
branch="master"
user="jenkins"
home_dir="/var/lib/jenkins/"
project_name="crosscat"

# print script usage
usage() {
    cat <<EOF
usage: $0 options
    
    Pull down repo from github and run tests via jenkins
    
    OPTIONS:
    -h      Show this message
    -p	    project name (default=$project_name)
    -b      set the branch to run (default=$branch)
    -u      set the user to install virtualenv under (default=$user)
    -h      set the dir to install virtualenv under (default=$home_dir)
EOF
exit
}


#Process command line arguments
while getopts hp:b:u:d: opt
do
    case "$opt" in
        h) usage;;
        p) project_name=$OPTARG;;
        b) branch=$OPTARG;;
	u) user=$OPTARG;;
	d) home_dir=$OPTARG;;
    esac
done

echo "jenkins_script.sh:"
echo "project_name: $project_name"
echo "user: $user"
echo "branch: $branch"
echo "home_dir: $home_dir"

cd $(dirname $(git rev-parse --git-dir))
python setup.py install
cd crosscat/tests/
PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH python /usr/local/bin/nosetests --with-xunit test_sampler_enumeration.py
exit

# Remove old source, and checkout newest source from master.
source /var/lib/jenkins/.bashrc
rm -rf $project_name
git clone --depth=1 https://github.com/mit-probabilistic-computing-project/$project_name.git $project_name --branch $branch
cd $project_name

# If the virtualenv isn't set up, do that.
if [ ! -e /var/lib/jenkins/.virtualenvs ]
then
  bash -i install_scripts/virtualenv_setup.sh $user $home_dir >virtualenv.out 2>virtualenv.err
  source /var/lib/jenkins/.bashrc
fi
workon $project_name

# Build and run tests. WORKSPACE is set by jenkins to /var/
export PYTHONPATH=$PYTHONPATH:$WORKSPACE
cd $WORKSPACE/$project_name
make tests
make cython
cd $project_name/tests
python /usr/bin/nosetests --with-xunit cpp_unit_tests.py cpp_long_tests.py test_sampler_enumeration.py
