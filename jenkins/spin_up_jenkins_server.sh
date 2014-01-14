set -e


# modifiable setings
local_crosscat_dir=/opt/crosscat
cluster_name=crosscat
if [[ ! -z $1 ]]; then
	cluster_name=crosscat
fi


# spin up the cluster
starcluster start -c crosscat -i c1.xlarge -s 1 $cluster_name
hostname=$(starcluster listclusters $cluster_name | grep master | awk '{print $NF}')
# open up the port for jenkins
local_jenkins_dir=$local_crosscat_dir/jenkins
starcluster shell < <(perl -pe "s/'crosscat'/'$cluster_name'/" $local_jenkins_dir/open_master_port_via_starcluster_shell.py)
# set up jenkins
starcluster sshmaster $cluster_name bash crosscat/jenkins/setup_jenkins.sh
# push up jenkins configuration
cd $local_jenkins_dir
python jenkins_utils.py --base_url http://$hostname:8080 -create


# notify user what hostname is
echo $hostname
