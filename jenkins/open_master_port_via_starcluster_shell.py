# USAGE: starcluster shell < open_master_port_via_starcluster.py

# settings
port = 8080
service_name = 'jenkins'
cluster_name = 'crosscat'
protocol = 'tcp'
world_cidr = '0.0.0.0/0'

master = cm.get_cluster(cluster_name).master_node
group = master.cluster_groups[0]

if isinstance(port, tuple):
    port_min, port_max = port
    pass
else:
    port_min, port_max = port, port
    pass

def get_port_open():
    # refresh master and group; not sure if necessary
    master = cm.get_cluster(cluster_name).master_node
    group = master.cluster_groups[0]
    #
    return master.ec2.has_permission(group, protocol, port_min, port_max, world_cidr)

if not get_port_open():
    import time as imported_time
    log_str = "Authorizing tcp ports [%s-%s] on %s for: %s" % \
            (port_min, port_max, world_cidr, service_name)
    print log_str
    master.ec2.conn.authorize_security_group(
            group_id=group.id, ip_protocol='tcp',
            from_port=port_min,
            to_port=port_max, cidr_ip=world_cidr)
    while not get_port_open():
        imported_time.sleep(2)

