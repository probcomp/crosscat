workon crosscat
pkill -f python\.\*/server_jsonrpc.py
nohup python -u crosscat/jsonrpc_http/server_jsonrpc.py >server_jsonrpc.out 2>server_jsonrpc.err &
