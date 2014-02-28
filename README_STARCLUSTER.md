CrossCat
==============

This package is configured to be installed as a StarCluster plugin.  Roughly, the following are prerequisites.

* An [Amazon EC2](http://aws.amazon.com/ec2/) account
    * [EC2 key pair](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/generating-a-keypair.html)
* [StarCluster](http://star.mit.edu/cluster/) installed on your local machine
    * ~/.starcluster/config file includes this repo's [starcluster.config](https://github.com/mit-probabilistic-computing-project/crosscat/blob/master/starcluster.config) by including the following line in the [global] section

     INCLUDE=/path/to/crosscat/starcluster.config
* You are able to start a 'smallcluster' cluster as defined in the default StarCluster config file
    * Make sure to fill in your credentials **and** have a properly defined keypair

     AWS_ACCESS_KEY_ID = #your_aws_access_key_id
     
     AWS_SECRET_ACCESS_KEY = #your_secret_access_key
     
     AWS_USER_ID= #your userid
     
     KEYNAME = mykey

    * To generate the default StarCluster config file, run

     starcluster -c [NONEXISTANT_FILE] help

A starcluster_plugin.py file in included in this repo.  Assuming the above prerequisites are fulfilled,

    local> starcluster start -s 1 -c crosscat [CLUSTER_NAME]

should start a single c1.medium StarCluster server on EC2, install the necessary software and compile the engine.

Everything will be set up for a user named 'crosscat'.  Required python packages will be installed to the system python.


Starting the engine
---------------------------
    local> starcluster sshmaster [CLUSTER_NAME] -u crosscat
    crosscat> bash /path/to/crosscat/scripts/service_scripts/run_server.sh
    crosscat> # test with 'python test_engine.py'

Setting up password login via ssh
---------------------------------
    local> starcluster sshmaster [CLUSTER_NAME]
    root> bash /home/crosscat/scripts/install_scripts/setup_password_login.sh -p <PASSWORD>

## [Creating an AMI](http://docs.aws.amazon.com/AWSEC2/latest/CommandLineReference/ApiReference-cmd-CreateImage.html) from booted instance

* Determine the instance id of the instance you want to create an AMI from.
   * You can list all instances with
    
    starcluster listinstances
    
* make sure you have your private key and X.509 certificate
   * your private key file, PRIVATE_KEY_FILE below, usually looks like pk-\<NUMBERS\_AND\_LETTERS\>.pem
   * your X.509 certificate file, CERT_FILE below, usually looks like cert-\<NUMBERS\_AND\_LETTERS\>.pem

Note, this will temporarily shut down the instance

    local> nohup ec2cim <instance-id> [--name <NAME>] [-d <DESCRIPTION>] -K ~/.ssh/<PRIVATE_KEY_FILE> -C ~/.ssh/<CERT_FILE> >out 2> err


This will start the process of creating the AMI.  It will print 'IMAGE [AMI-NAME]' to the file 'out'.  Record AMI-NAME and modify ~/.starcluster/config to use that for the crosscat cluster's NODE\_IMAGE\_ID.

<!---
Caching HTTPS password
----------------------
When a StarCluster machine is spun up, its .git origin is changed to the github https address.  You can perform git operations but github repo operations will require a password.  You can cache the password by performing the following operations (from the related github [help page](https://help.github.com/articles/set-up-git#password-caching))

     crosscat> git config --global credential.helper cache
     crosscat> git config --global credential.helper 'cache --timeout=3600'

This requires git 1.7.10 or higher.  To get on ubuntu, do
sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update
sudo apt-get install -y git
--->
