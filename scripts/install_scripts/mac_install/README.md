# Mac OSX installation

**Intended for CrossCat development only. The virtual machine is the preferred method of installation.**

## Notes
- Tested on OSX 10.7, 10.8, and 10.9
- Requires XCode with command line tools
- Requires [Homebrew](http://brew.sh/)


## System impact
- Installs CrossCat in a python [virtual environmentt](https://pypi.python.org/pypi/virtualenv) which is an isolated development environmental with its own modules (so it won't mess with your system modules)
- Adds the following environmental variables to `~/.bash_profile`
 - adds crosscat repo directory to `PYTHONPATH`
 - adds the virtual environment variable `WORKON_HOME`
 - adds `CROSSCAT_BOOST_ROOT` which tells the crosscat makefile where [boost](http://www.boost.org/) is installed


## Installation Instructions

Make the installation script executable

     $ chmod +x install_mac.sh

Run the script.

     $ ./install_mac.sh

 **Note:** The installation takes a while to complete. You will be prompted for your password multiple times.
 
The script installs all python dependencies in a virtual environment called `crosscat`. If the installer detects an existing `crosscat` virtual environment, you have the option to delete and reinstall or to create a new virtual environment. 

##Recompiling the Cython/C++ code

The install script will compile the code for you, but if you wish to compile it yourself there is a makefile in `crosscat/cython_code` that will allow you to do so. Run it with the following command

	$ make -f Makefile.mac

The makefile requires that the environmental variable `CROSSCAT_BOOST_ROOT` exists and is set to the path that houses your Boost headers. If you have run the install script, you are set; the following line (or similar) has been added to your `~/.bash_profile`:

	export CROSSCAT_BOOST_ROOT=/usr/local/Cellar/boost/1.54.0/include

If you run the makefile and are told that everything is up-to-date run

	$ make -f Makefile.mac clean

to remove the compiled files and then run the makefile to compile.

## Running code

You will need to either source your `bash_profile`

	$ source ~/.bash_profile	

or reopen terminal. Then enter your virtual environment

	$ workon crosscat

At this point you should be good to go.