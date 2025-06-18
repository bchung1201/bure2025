import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import deque

inputRates = np.array([0.2, 0.2])
processRates = np.array([0.3, .5])

# Define accessible servers for each queue
accessibleServers = [[0], [0,1]]

def max_flow(inputRates, processRates, accessibleServers):
    """
    Find maximum flow through router-server network using Edmonds-Karp algorithm.
    
    Network structure:
    - Node 0: Synthetic source
    - Nodes 1 to len(inputRates): Routers
    - Nodes len(inputRates)+1 to len(inputRates)+len(processRates): Servers
    - Node len(inputRates)+len(processRates)+1: Synthetic sink
    """
    n_routers = len(inputRates)
    n_servers = len(processRates)
    n_nodes = n_routers + n_servers + 2  # +2 for source and sink
    
    source = 0
    sink = n_nodes - 1
    
    # Initialize capacity matrix
    capacity = np.zeros((n_nodes, n_nodes))
    
    # Connect source to routers
    for i in range(n_routers):
        router_node = i + 1
        capacity[source][router_node] = inputRates[i]
    
    # Connect routers to accessible servers (infinite capacity)
    for router_idx, servers in enumerate(accessibleServers):
        router_node = router_idx + 1
        for server_idx in servers:
            server_node = n_routers + 1 + server_idx
            capacity[router_node][server_node] = float('inf')
    
    # Connect servers to sink
    for i in range(n_servers):
        server_node = n_routers + 1 + i
        capacity[server_node][sink] = processRates[i]
    
    def bfs_find_path(source, sink, parent):
        """Find augmenting path using BFS"""
        visited = np.zeros(n_nodes, dtype=bool)
        queue = deque([source])
        visited[source] = True
        
        while queue:
            u = queue.popleft()
            
            for v in range(n_nodes):
                if not visited[v] and capacity[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False
    
    parent = np.full(n_nodes, -1)
    max_flow_value = 0
    
    # Find augmenting paths and update flow
    while bfs_find_path(source, sink, parent):
        # Find minimum capacity along the path
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, capacity[parent[s]][s])
            s = parent[s]
        
        # Add path flow to overall flow
        max_flow_value += path_flow
        
        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = parent[v]
    
    return max_flow_value

# Calculate and print the maximum flow
max_flow = max_flow(inputRates, processRates, accessibleServers)
print(f"Maximum flow through the network: {max_flow}")
print(f"Input rates: {inputRates}")
print(f"Process rates: {processRates}")
print(f"Accessible servers: {accessibleServers}")

