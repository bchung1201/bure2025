import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit

T = 150000
rate = 1 / math.sqrt(T)
gamma = rate

sample = 200
stepsize = 0.01
numQueues = 2
numServers = 2
M = 20

inputRates = np.array([0.28, 0.28])
processRates = np.array([0.6, 0.2])

# Define accessible servers for each queue
accessibleServers = [[0], [0, 1]]

# Pre-compute mapping for optimization
max_accessible = max(len(servers) for servers in accessibleServers)
accessible_matrix = np.full((numQueues, max_accessible), -1, dtype=np.int32)
accessible_lengths = np.zeros(numQueues, dtype=np.int32)

for q in range(numQueues):
    accessible_lengths[q] = len(accessibleServers[q])
    for i, server in enumerate(accessibleServers[q]):
        accessible_matrix[q, i] = server

# Maximum queue size for pre-allocation (adjust if needed)
MAX_QUEUE_SIZE = 1000

@jit(nopython=True)
def sample_from_weights(weights, random_val):
    """Fast weighted sampling using pre-generated random number"""
    cumsum = np.cumsum(weights)
    if cumsum[-1] == 0:
        return np.random.randint(len(weights))
    return np.searchsorted(cumsum, random_val * cumsum[-1])

@jit(nopython=True)
def find_oldest_packet_index(timestamps, valid_count):
    """Find index of oldest packet among valid packets"""
    if valid_count == 0:
        return -1
    oldest_idx = 0
    oldest_time = timestamps[0]
    for i in range(1, valid_count):
        if timestamps[i] < oldest_time:
            oldest_time = timestamps[i]
            oldest_idx = i
    return oldest_idx

@jit(nopython=True)
def run_bipartite_fifo_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                                  arrivals, noise_choices, weight_randoms, processed_batch,
                                  accessible_matrix, accessible_lengths):
    """Optimized FIFO bipartite simulation with JIT compilation"""
    
    # Initialize weights - different sizes for each queue
    weights = np.zeros((numQueues, max_accessible))
    for q in range(numQueues):
        for i in range(accessible_lengths[q]):
            weights[q, i] = 1.0 / accessible_lengths[q]
    
    # FIFO queue data structures
    # For each queue: timestamps and queue_counts
    queue_timestamps = np.full((numQueues, MAX_QUEUE_SIZE), -1, dtype=np.int32)
    queue_counts = np.zeros(numQueues, dtype=np.int32)
    
    # Server buffers: store timestamp of packet in buffer (-1 if empty)
    buffer_timestamps = np.full(numServers, -1, dtype=np.int32)
    buffer_queue_ids = np.full(numServers, -1, dtype=np.int32)
    
    for t in range(T):
        # Arrivals - add to end of FIFO queues
        for q in range(numQueues):
            if arrivals[q, t] == 1:
                if queue_counts[q] < MAX_QUEUE_SIZE:
                    queue_timestamps[q, queue_counts[q]] = t
                    queue_counts[q] += 1
        
        # Server selection for active queues
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        costs = np.zeros((numQueues, max_accessible))
        
        # Packets sent to servers this timestep
        server_packet_timestamps = np.full((numServers, numQueues), -1, dtype=np.int32)
        server_packet_counts = np.zeros(numServers, dtype=np.int32)
        
        for q in range(numQueues):
            if queue_counts[q] > 0:
                accessible_count = accessible_lengths[q]
                
                if noise_choices[q, t]:
                    # Random choice from accessible servers
                    server_idx = int(weight_randoms[q, t] * accessible_count)
                    chosen_servers[q] = accessible_matrix[q, server_idx]
                else:
                    # Weighted choice from accessible servers
                    active_weights = weights[q, :accessible_count]
                    server_idx = sample_from_weights(active_weights, weight_randoms[q, t])
                    chosen_servers[q] = accessible_matrix[q, server_idx]
                
                server = chosen_servers[q]
                # Send oldest packet (FIFO) - packet at index 0
                oldest_timestamp = queue_timestamps[q, 0]
                server_packet_timestamps[server, server_packet_counts[server]] = oldest_timestamp
                server_packet_counts[server] += 1
                
                # Remove oldest packet from queue (shift array left)
                for i in range(queue_counts[q] - 1):
                    queue_timestamps[q, i] = queue_timestamps[q, i + 1]
                queue_counts[q] -= 1
        
        # Use pre-generated processing outcomes
        processed = processed_batch[:, t]
        
        for k in range(numServers):
            # Handle buffer logic
            if server_packet_counts[k] > 0:
                if buffer_timestamps[k] == -1:  # Buffer empty
                    # Find oldest packet among all packets sent to this server
                    oldest_idx = find_oldest_packet_index(
                        server_packet_timestamps[k, :server_packet_counts[k]], 
                        server_packet_counts[k]
                    )
                    oldest_timestamp = server_packet_timestamps[k, oldest_idx]
                    
                    # Find which queue this packet came from
                    chosen_queue = -1
                    for q in range(numQueues):
                        if chosen_servers[q] == k:
                            chosen_queue = q
                            break
                    
                    buffer_timestamps[k] = oldest_timestamp
                    buffer_queue_ids[k] = chosen_queue
                    
                    # Update weights
                    if chosen_queue != -1:
                        for j in range(accessible_lengths[chosen_queue]):
                            if accessible_matrix[chosen_queue, j] == k:
                                costs[chosen_queue, j] = -1.0
                                # Update weights
                                cost = costs[chosen_queue, j] / ((gamma / accessible_lengths[chosen_queue]) + 
                                                               ((1 - gamma) * weights[chosen_queue, j]))
                                weights[chosen_queue, j] *= np.exp((gamma / accessible_lengths[chosen_queue]) * (-cost))
                                break
            
            # Process buffer if packet present
            if buffer_timestamps[k] != -1 and processed[k] == 1:
                buffer_timestamps[k] = -1
                buffer_queue_ids[k] = -1
        
        # Normalize weights for all queues
        for q in range(numQueues):
            weight_sum = 0.0
            for j in range(accessible_lengths[q]):
                weight_sum += weights[q, j]
            if weight_sum > 0:
                for j in range(accessible_lengths[q]):
                    weights[q, j] /= weight_sum
    
    return np.sum(queue_counts)

##################
###WITH BUFFERS###
##################
print("Starting optimized FIFO bipartite simulation...")
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)

for m in range(M):
    print(f"Reached m: {m}")
    inputRates += 0.01
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)
    
    # Pre-generate random numbers for this ratio
    arrivals_batch = np.random.binomial(1, inputRates[:, np.newaxis, np.newaxis], 
                                       size=(numQueues, sample, T))
    noise_choices_batch = np.random.binomial(1, gamma, size=(numQueues, sample, T))
    weight_randoms_batch = np.random.random(size=(numQueues, sample, T))
    processed_batch = np.random.binomial(1, processRates[:, np.newaxis, np.newaxis], 
                                        size=(numServers, sample, T))
    
    for r in range(sample):
        # Extract pre-generated randoms for this sample
        arrivals = arrivals_batch[:, r, :]
        noise_choices = noise_choices_batch[:, r, :]
        weight_randoms = weight_randoms_batch[:, r, :]
        processed = processed_batch[:, r, :]
        
        sumBuildup = run_bipartite_fifo_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            arrivals, noise_choices, weight_randoms, processed,
            accessible_matrix, accessible_lengths
        )
        
        buildup[r] = sumBuildup / (T * numQueues)

    avgBuildup[m] = np.mean(buildup)
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)

print("Simulation complete! Generating plot...")

# Generate Figure showing buildup in a bipartite system
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='blue')
ax2.axvline(x=1/3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
plt.title('Bipartite Graph EXP3 - FIFO Optimized')
plt.show()