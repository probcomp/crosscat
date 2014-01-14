#!/bin/bash


# set default values
jenkins_home=/var/lib/jenkins/
user=crosscat
project_name=crosscat


# print script usage
usage() {
	cat <<EOF
useage: $0 options

	Set up jenkins

	OPTIONS:
	-h	Show this message
	-p	project_name=$project_name
	-u	user=$user
	-j	jenkins_home=$jenkins_home
EOF
exit
}

# Process the arguments
while getopts hp:u:j: opt
do
	case "$opt" in
		h) usage;;
		p) project_name=$OPTARG;;
		u) user=$OPTARG;;
		j) jenkins_home=$OPTARG;;
	esac
done


# Helper functions
function install_jenkins_plugin () {
	plugin_name=$1
	plugin_version=$2
	base_url=http://updates.jenkins-ci.org/download/plugins/
	full_url=$base_url/$plugin_name/$plugin_version/${plugin_name}.hpi
	jar_dir=/var/cache/jenkins/war/WEB-INF/
	plugin_dir=${jar_dir}/plugins/
	#
	rm ${plugin_dir}/${plugin_name}.hpi
	java -jar ${jar_dir}/jenkins-cli.jar -s http://127.0.0.1:8080/ install-plugin $full_url
}

function restart_jenkins () {
	jar_dir=/var/cache/jenkins/war/WEB-INF/
	java -jar ${jar_dir}/jenkins-cli.jar -s http://127.0.0.1:8080/ safe-restart
}

function wait_for_jenkins_to_respond () {
	wget http://127.0.0.1:8080 2>/dev/null 1>/dev/null
	while [ $? -ne "0" ]
	do
		sleep 2
		wget http://127.0.0.1:8080 2>/dev/null 1>/dev/null
	done
}


# install jenkins
#   per http://pkg.jenkins-ci.org/debian-stable/
wget -q -O - http://pkg.jenkins-ci.org/debian-stable/jenkins-ci.org.key | sudo apt-key add -
sudo echo "deb http://pkg.jenkins-ci.org/debian-stable binary/" >> /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y jenkins
sudo apt-get update
#
# make sure jenkins api available for job setup automation
pip install jenkinsapi==0.1.13

# run some helper scripts
# set up headless matplotlib
mkdir -p ${jenkins_home}/.matplotlib
echo backend: Agg > ${jenkins_home}/.matplotlib/matplotlibrc
chown -R jenkins $jenkins_home
# set up password login, set password for jenkins user
bash crosscat/scripts/install_scripts/setup_password_login.sh -u jenkins -p bigdata

# make sure jenkins can install python packages
python_dir=/usr/local/lib/python2.7/dist-packages
chown -R jenkins $python_dir


# wait for jenkins to respond before proceeding
wait_for_jenkins_to_respond

#  install plugins and restart
install_jenkins_plugin scm-api 0.1
install_jenkins_plugin credentials 1.9.3
install_jenkins_plugin ssh-credentials 1.5.1
install_jenkins_plugin git-client 1.6.0
install_jenkins_plugin git 2.0.1
install_jenkins_plugin github-api 1.44
install_jenkins_plugin github 1.8
restart_jenkins
