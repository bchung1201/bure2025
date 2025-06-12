import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit
from numba.typed import List

T = 150000
rate = 1 / math.sqrt(T)
gamma = rate

sample = 200
stepsize = 0.01
numQueues = 2
numServers = 2
M = 20

inputRates = np.array([0.24, 0.24])
processRates = np.array([0.6, 0.2])

# Create typed list for numba - need to specify type since lists are empty
from numba import types
queues = List()
for i in range(numQueues):
    queues.append(List.empty_list(types.int32))

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

@jit(nopython=True)
def sample_from_weights(weights, random_val):
    """Fast weighted sampling using pre-generated random number"""
    cumsum = np.cumsum(weights)
    if cumsum[-1] == 0:
        return np.random.randint(len(weights))
    return np.searchsorted(cumsum, random_val * cumsum[-1])

@jit(nopython=True) 
def run_bipartite_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                            noise_choices, weight_randoms,
                            accessible_matrix, accessible_lengths, queues):
    """Optimized bipartite simulation with JIT compilation"""
    
    # Initialize weights - different sizes for each queue
    weights = np.zeros((numQueues, max_accessible))
    for q in range(numQueues):
        for i in range(accessible_lengths[q]):
            weights[q, i] = 1.0 / accessible_lengths[q]

    buffers = np.full(numServers, -1, dtype=np.int32)
    
    # Reset queues at start of each simulation
    for q in range(numQueues):
        queues[q].clear()
    
    for t in range(T):
        # Arrivals
        for q in range(numQueues):
            rng = np.random.binomial(1, inputRates[q])
            if rng == 1:
                queues[q].append(t)
        
        # Server selection for active queues
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        costs = np.zeros((numQueues, max_accessible))
        
        for q in range(numQueues):
            if len(queues[q]) > 0:
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
        
        for k in range(numServers):
            # Find which queues sent packets to this server
            sending_queues = []
            for q in range(numQueues):
                if chosen_servers[q] == k:
                    sending_queues.append(q)
            
            # Handle buffer logic
            chosen_queue = -1
            if len(sending_queues) > 0:
                if buffers[k] == -1:  # Buffer empty
                    # Randomly accept one packet
                    oldest_queues = []
                    min_time = T+1
                    for q in sending_queues:
                        min_time = min(min_time, queues[q][0])
                    for q in sending_queues:
                        if queues[q][0] == min_time:
                            oldest_queues.append(q)
                    idx = np.random.randint(len(oldest_queues))
                    chosen_queue = oldest_queues[idx]
                    buffers[k] = chosen_queue
                    
                    # FIXED: Correct weight update logic
                    # Find which index in the accessible servers corresponds to server k
                    for j in range(accessible_lengths[chosen_queue]):
                        if accessible_matrix[chosen_queue, j] == k:
                            costs[chosen_queue, j] = -1.0
                            # Update weights
                            cost = costs[chosen_queue, j] / ((gamma / accessible_lengths[chosen_queue]) + 
                                                           ((1 - gamma) * weights[chosen_queue, j]))
                            weights[chosen_queue, j] *= np.exp((gamma / accessible_lengths[chosen_queue]) * (-cost))
                            break
            
            # Process buffer if packet present
            if buffers[k] != -1 and np.random.binomial(1, processRates[k]) == 1:
                buffers[k] = -1
            
            # Remove packet if it was accepted
            if chosen_queue != -1:
                queues[chosen_queue].pop(0)
        
        # Normalize weights for all queues
        for q in range(numQueues):
            weight_sum = 0.0
            for j in range(accessible_lengths[q]):
                weight_sum += weights[q, j]
            if weight_sum > 0:
                for j in range(accessible_lengths[q]):
                    weights[q, j] /= weight_sum
    res = 0
    for q in range(numQueues):
        res += len(queues[q])
    return res

##################
###WITH BUFFERS###
##################
print("Starting optimized bipartite simulation...")
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
    noise_choices_batch = np.random.binomial(1, gamma, size=(numQueues, sample, T))
    weight_randoms_batch = np.random.random(size=(numQueues, sample, T))
    
    for r in range(sample):
        # Extract pre-generated randoms for this sample
        noise_choices = noise_choices_batch[:, r, :]
        weight_randoms = weight_randoms_batch[:, r, :]
        
        sumBuildup = run_bipartite_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            noise_choices, weight_randoms,
            accessible_matrix, accessible_lengths, queues
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
plt.title('Bipartite Graph EXP3 - Optimized')
plt.show()