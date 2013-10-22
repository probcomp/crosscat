# must NOT be root!
if [[ $(whoami) == "root" ]]; then
	echo "$0: Can't be root to run!"
	exit
fi


# get rvm
\curl -L https://get.rvm.io | bash
source ~/.rvm/scripts/rvm
# get the right ruby
rvm install ruby-1.9.3-p448
