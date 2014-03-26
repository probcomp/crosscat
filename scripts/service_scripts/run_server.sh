script_name=server_jsonrpc.py
#
this_script_path=$(readlink -f "$0")
this_script_dir=$(dirname $this_script_path)
this_git_dir=$(cd $this_script_dir && git rev-parse --git-dir)
this_repo_dir=$(dirname $this_git_dir)
#
script_path=$this_repo_dir/crosscat/jsonrpc_http/$script_name


cd $this_script_dir
pkill -f python\.\*/$script_name
nohup python -u $script_path >${script_path%.*}.out 2>${script_path%.*}.err &
