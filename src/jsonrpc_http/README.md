crosscat/jsonrpc_http
==============

crosscat JSONRPCEngine stub server

Running tests
---------------------------
    dirname=/path/to/crosscat
    # Start the server
    bash $dirname/scripts/service_scripts/run_server.sh
    cd $dirname/crosscat/jsonrpc_http
    # this is currently broken!
    python stub_client_jsonrpc.py >stub_client_jsonrpc.out 2>stub_client_jsonrpc.err
    python test_engine.py >test_engine.out 2>test_engine.err
    # Kill the server
    pkill -f server_jsonrpc
