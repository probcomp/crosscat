#!/bin/bash
set -e
set -v


# set default values
jenkins_home=/var/lib/jenkins/
jar_dir=/var/cache/jenkins/war/WEB-INF/
jenkins_uri=http://127.0.0.1:8080/


# Helper functions
function install_jenkins_plugin () {
	plugin_name=$1
	plugin_version=$2
	base_url=http://updates.jenkins-ci.org/download/plugins/
	full_url=$base_url/$plugin_name/$plugin_version/${plugin_name}.hpi
	plugin_dir=${jar_dir}/plugins/
	#
	rm ${plugin_dir}/${plugin_name}.hpi 2>/dev/null || true
	java -jar ${jar_dir}/jenkins-cli.jar -s $jenkins_uri install-plugin $full_url
	return 0
}

function restart_jenkins () {
	java -jar ${jar_dir}/jenkins-cli.jar -s $jenkins_uri safe-restart || true
	return 0
}

function wait_for_web_response () {
	set +e
	uri=$1
	wget $uri 2>/dev/null 1>/dev/null
	while [ $? -ne "0" ]
	do
		sleep 2
		wget $uri 2>/dev/null 1>/dev/null
	done
	set -e
	return 0
}


# install jenkins
#   per http://pkg.jenkins-ci.org/debian-stable/
wget -q -O - http://pkg.jenkins-ci.org/debian-stable/jenkins-ci.org.key | sudo apt-key add -
sudo echo "deb http://pkg.jenkins-ci.org/debian-stable binary/" >> /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y jenkins
#
# make sure jenkins api available for job setup automation
pip install jenkinsapi==0.1.13


# install plugins and restart
#
# wait for jenkins to respond before installing
wait_for_web_response $jenkins_uri
#
install_jenkins_plugin scm-api 0.1
install_jenkins_plugin credentials 1.9.3
install_jenkins_plugin ssh-credentials 1.5.1
install_jenkins_plugin git-client 1.6.0
install_jenkins_plugin git 2.0.1
install_jenkins_plugin github-api 1.44
install_jenkins_plugin github 1.8
restart_jenkins
#
# wait for jenkins to respond before returning
wait_for_web_response $jenkins_uri


# miscellaneous support operations
#
# set up headless matplotlib
mkdir -p ${jenkins_home}/.matplotlib
echo backend: Agg > ${jenkins_home}/.matplotlib/matplotlibrc
chown -R jenkins $jenkins_home
#
# set up password login, set password for jenkins user
bash crosscat/scripts/install_scripts/setup_password_login.sh -u jenkins -p bigdata
#
# make sure jenkins can install python packages
python_dir=/usr/local/lib/python2.7/dist-packages
chown -R jenkins $python_dir
