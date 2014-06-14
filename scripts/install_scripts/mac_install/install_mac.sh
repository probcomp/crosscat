#!/bin/bash

################################################################################
#						   CHECK INSTALL REQUIREMENTS
################################################################################
# Check for OSX version
OSX_VERSION=`sw_vers -productVersion | grep -o '[0-9]' | awk '{i++}i==3'`
if [ "$OSX_VERSION" -lt 7 ]
then
	echo "This script does not currently work on OSX below 10.7 (Lion)."
	exit
fi

# check for xcode and command line tools
if [ -z `which xcodebuild` ]
then
	echo "Installation requires XCode."	
	echo "Please install XCode from the App Store."
	echo "Once you have installed XCode, please install the command line tools from XCode\'s preferences menu."
	exit
fi

# check for homebrew
if [ -z `which brew` ]
then
	echo "Installation requires Homebrew (http://brew.sh/)."
	exit
fi

################################################################################
#						   INSTALL REQUIREMENTS
################################################################################
set -e

sudo chown -R `whoami` /usr/local/lib/
sudo -k

# start installing dependencies
brew install libpng
brew install freetype
# brew install valgrind # not supported on Mac OS 10.9
# Scipy needs a fortran compiler
brew install gfortran
# Using python give access denied (don't want to use sudo)
brew install boost --without-python 

# cd into crosscat
cd ../../../
PROJECT_DIRECTORY=`pwd`
SHELL_FILE=$HOME/.bash_profile


if [ -z "$PYTHONPATH" ]
then
	if [ -z `cat ~/.bash_profile|grep "export CROSSCAT_BOOST_ROOT"` ]
	then
		# if there is no PYTHONPATH we will need to fill it in
		echo "Creating PYTHONPATH."
		echo "" >> $SHELL_FILE
		echo "# Python will search these directories for modules" >> $SHELL_FILE
		echo "export PYTHONPATH=$PROJECT_DIRECTORY" >> $SHELL_FILE
	fi
else
	# check if the python path is set up correctly
	if [[ "$PYTHONPATH" =~ (^|:)"${PROJECT_DIRECTORY}"(:|$) ]]
	then
		# if the project directory is in PYTHONPATH, we're fine
		echo "Python path is good to go."
	else
		# if the project directory is not in the python
		echo "Adding ${PROJECT_DIRECTORY} to PYTHONPATH in $SHELL_FILE." 
		# write the new python path
		NEW_PYTHON_PATH=${PROJECT_DIRECTORY}:$PYTHONPATH
		sed "s/${PYTHONPATH}/c\${NEW_PYTHON_PATH}" $SHELL_FILE
	fi
fi

source $SHELL_FILE

################################################################################
#						   VIRTUAL ENVIRONMENT SETUP
################################################################################
# if no $WORKON_HOME then install virtual environment and set up the bash_profile
if [ -z "$WORKON_HOME" ]
then
	
	pip install virtualenv && pip install virtualenvwrapper 
	if [ $? = 1 ]
	then
		echo Failed.
		echo "Installation of virtualenv failed."
		exit
	fi
	
	virtualenv_home="$HOME/.virtualenvs"
	
	# add WORKON_HOME to .bash_profile
	echo "Adding WORKON_HOME=$virtualenv_home to bash_profile."
	echo "" >> $SHELL_FILE
	echo "# Virtual environment home. Directory where the vitual environments are stored." >> $SHELL_FILE
	echo "export WORKON_HOME=$virtualenv_home" >> $SHELL_FILE
	
	# if virtualenvwrapper.sh is not sources in bash_profile, source it
	source_wrapper_string="source /usr/local/bin/virtualenvwrapper.sh"
	line_number=`awk '$0 ~ str{print NR-1 FS b}{b=$0}' str="$source_wrapper_string" $SHELL_FILE` 
	
	if [ -z "$line_number" ]
	then
		echo "" >> $SHELL_FILE
		echo "# shell commands for virtual environment" >> $SHELL_FILE
		echo $source_wrapper_string >> $SHELL_FILE
	fi
fi

# source the shell file so we can begin to use the wrapper commands
source $SHELL_FILE

echo "**Did things set?**"
echo "$WORKON_HOME"
echo "$PYTHONPATH"
echo `lsvirtualenv  -b`

# check if the crosscat virtualenv already exists
# exists=`lsvirtualenv  -b | grep "^crosscat$"`
if [ -z `lsvirtualenv  -b | grep "^crosscat$"` ]
then
	virtual_env_name="crosscat"
else
	echo "You appear to have an existing crosscat virtual environment."
	echo "Would you like to create a new virtual environment, or delete and reinstall the old one?"
	echo " "
	echo "1. New virtual environment."
	echo "2. Delete and reinstall."
	echo "3. Exit."
	echo " "
	echo -ne "Enter choice: "

	read user_input
	
	case "$user_input" in
		1) 
			echo ""
			echo -ne "Enter the name of the new virtual environment: "
			read virtual_env_name
			;;
		2) 
			echo "Deleting crosscat virtal environment."
			sudo rm -rf $WORKON_HOME/crosscat
			sudo -k
			virtual_env_name="crosscat"
			;;
		*) 
			echo "Exiting."
			exit
			;;
	esac
fi

echo " "

set +e

# build the virtual environment and switch to it
mkvirtualenv $virtual_env_name -a $PROJECT_DIRECTORY
workon $virtual_env_name

if [ -z "$VIRTUAL_ENV" ]
then
	# something went wrong and we're not in a virtualenv
	echo "not in virtualenv"
	exit
fi

set -e

################################################################################
#						       PYTHON REQUIREMENTS
################################################################################
# event though numpy is in the requirements file, Matplotlib depends on a full,
# working install of numpy. Because pip makes 2 runs through requirements files
# the numpy install is not actually complete when it gets to install matplotlib
# so this is hack to fix that problem.
pip install numpy==1.7.0
pip install -r requirements.txt

################################################################################
#						       COMPILING CYTHON CODE
################################################################################
cd $PROJECT_DIRECTORY/crosscat/cython_code

# add the BOOST_INCLUDE directory to bash_profile so the user can compile the
# cython code without having to go through the whole install process
if [ -z "$CROSSCAT_BOOST_ROOT" ]
then
	# get the boost include directory
	BOOST_INCLUDE=`brew ls boost|grep "include"|head -n1|grep -o '/.*include'`

	if [ -z "$BOOST_INCLUDE" ]
	then
		echo "Could not locate your boost/include dierectory from homebrew."
		exit
	fi
	
	# check if CROSSCAT_BOOST_ROOT is already set in bash_profile
	if [ -z `cat ~/.bash_profile|grep "export CROSSCAT_BOOST_ROOT"` ]
	then
		echo "adding $BOOST_INCLUDE to bash_profile"
		echo "" >> $SHELL_FILE
		echo "# boost include directory for crosscat" >> $SHELL_FILE
		echo "export CROSSCAT_BOOST_ROOT=$BOOST_INCLUDE" >> $SHELL_FILE
	fi	
	
	source $SHELL_FILE
fi
	
# build the cython code
python setup.py build_ext --inplace -I$CROSSCAT_BOOST_ROOT 

# The end