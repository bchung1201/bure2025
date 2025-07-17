import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import deque

inputRates = np.array([0.1, 0.5, 0.7])
processRates = np.array([0.3, 0.9, 0.6])

# Define accessible servers for each queue
accessibleServers = [[0, 1], [0, 2], [1]]

ratio = 0

#x * sum_A lambda_i <= sum_neighbors(A) mu_i
#want to minimize x across cuts.
#bsearch on 1/x
def binary_search():
    def is_possible(x):
        scaled_input_rates = x * inputRates
        flow = max_flow(scaled_input_rates, processRates, accessibleServers)
        total_input = np.sum(scaled_input_rates)
        epsilon = 0.00000001

        return flow + epsilon >= total_input
    
    #bsearch on 1/x
    left = 1e-6
    right = 1.0
    epsilon = 1e-8
    
    while right - left > epsilon:
        mid = (left + right) / 2
        x = 1.0 / mid
        
        if is_possible(x):
            right = mid  #x works, we can increase x, so 1/x shrinks
        else:
            left = mid   #opposite
    
    optimal_x = 1.0 / right
    return optimal_x


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
flow_result = max_flow(inputRates, processRates, accessibleServers)
print(f"Maximum flow through the network: {flow_result}")
print(f"Input rates: {inputRates}")
print(f"Process rates: {processRates}")
print(f"Accessible servers: {accessibleServers}")

# Find optimal x using binary search
optimal_x = binary_search()
print(f"\nOptimal x (maximum scaling factor): {optimal_x}")
print(f"This means the constraint x * sum_A lambda_i <= sum_neighbors(A) mu_i")
print(f"is satisfied for all cuts A when x <= {optimal_x}")

# Verify the result
scaled_rates = optimal_x * inputRates
scaled_flow = max_flow(scaled_rates, processRates, accessibleServers)
total_scaled_input = np.sum(scaled_rates)
print(f"\nVerification:")
print(f"Scaled input rates: {scaled_rates}")
print(f"Max flow with scaled rates: {scaled_flow}")
print(f"Total scaled input: {total_scaled_input}")
print(f"Flow >= Input: {scaled_flow >= total_scaled_input}")

print(max_flow(1.4*inputRates, processRates, accessibleServers))
