#!python
#
#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
import os
#
from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log


# maybe should prefix the command with "source /etc/profile"
# as starclusters' sshutils.ssh.execute(..., source_profile=True) does
def execute_as_user(node, user, command_str, **kwargs):
     cmd_str = 'sudo -H -u %s %s'
     cmd_str %= (user, command_str)
     node.ssh.execute(cmd_str, **kwargs)


class crosscatSetup(ClusterSetup):
     def __init__(self):
         # TODO: Could be generalized to "install a python package plugin"
         pass

     def run(self, nodes, master, user, user_shell, volumes):
          # NOTE: nodes includes master
         for node in nodes:
               log.info("Installing CrossCat as root on %s" % node.alias)
               #
               # FIXME: should be capturing out, err from script executions
               cmd_strs = [
                       # FIXME: could add an if [[ ! -d crosscat ]]; then ... done
                       # to squelch 'git clone' error messages
                       'rm -rf crosscat',
                       'git clone https://github.com/mit-probabilistic-computing-project/crosscat.git',
                       'bash crosscat/scripts/install_scripts/install.sh',
                       'python crosscat/setup.py install',
                       ]
               for cmd_str in cmd_strs:
                   node.ssh.execute(cmd_str)
         for node in nodes:
               log.info("Setting up CrossCat as user on %s" % node.alias)
               #
               cmd_strs = [
                   'mkdir -p ~/.matplotlib',
                   'echo backend: Agg > ~/.matplotlib/matplotlibrc',
                    ]
               for cmd_str in cmd_strs:
                   cmd_str = 'bash -c "source /etc/profile && %s"' % cmd_str
                   execute_as_user(node, user, cmd_str)
#               # run server
#               cmd_str = 'bash -i %s'
#               cmd_str %= S.path.run_server_script.replace(S.path.this_repo_dir, S.path.remote_code_dir)
#               run_as_user(node, user, cmd_str)
#               #
#               cmd_str = "bash -i %s %s" % (S.path.run_webserver_script.replace(S.path.this_repo_dir, S.path.remote_code_dir),
#                                            S.path.web_resources_dir.replace(S.path.this_repo_dir, S.path.remote_code_dir))
#               run_as_user(node, user, cmd_str)
