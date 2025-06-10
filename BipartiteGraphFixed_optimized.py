import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit

T = 50000
sample = 200
stepsize = 0.01
numQueues = 2
numServers = 2
M = 10

inputRates = np.array([0.30, 0.30])
processRates = np.array([0.6, 0.2])

# Define accessible servers for each queue
accessibleServers = [[0], [0, 1]]

@jit(nopython=True) 
def run_fixed_routing_simulation(inputRates, processRates, T, numQueues, numServers,
                                arrivals, routing_randoms, processed_batch):
    """Fixed routing simulation - no learning, just fixed probabilities"""
    
    queues = np.zeros(numQueues)
    buffers = np.full(numServers, -1, dtype=np.int32)
    
    for t in range(T):
        # Arrivals
        queues += arrivals[:, t]
        
        # Server selection with FIXED routing probabilities
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        
        # Queue 0: Can only access server 0, so always send there
        if queues[0] > 0:
            chosen_servers[0] = 0
        
        #Queue 1: Can access servers 0 and 1, send with 0.5 probability to each
        if queues[1] > 0:
            chosen_servers[1] = np.random.randint(0, 2)
        
        # Use pre-generated processing outcomes
        processed = processed_batch[:, t]
        chosen_packets = np.full(numServers, -1, dtype=np.int32)

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
                    idx = np.random.randint(len(sending_queues))
                    chosen_queue = sending_queues[idx]
                    buffers[k] = chosen_queue

                # Accept packet into buffer

            if len(sending_queues) > 0 and buffers[k] == -1:
                idx = np.random.randint(len(sending_queues))
                chosen_queue = sending_queues[idx]
                buffers[k] = chosen_queue

            # Process buffer and remove packet only if successful
            if buffers[k] != -1 and processed[k] == 1:
                queues[buffers[k]] -= 1  # Remove from queue when actually processed
                buffers[k] = -1


    return np.sum(queues)

##################
###FIXED ROUTING###
##################
print("Starting fixed routing simulation...")
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
    routing_randoms_batch = np.random.random(size=(numQueues, sample, T))
    processed_batch = np.random.binomial(1, processRates[:, np.newaxis, np.newaxis], 
                                        size=(numServers, sample, T))
    
    for r in range(sample):
        # Extract pre-generated randoms for this sample
        arrivals = arrivals_batch[:, r, :]
        routing_randoms = routing_randoms_batch[:, r, :]
        processed = processed_batch[:, r, :]
        
        sumBuildup = run_fixed_routing_simulation(
            inputRates, processRates, T, numQueues, numServers,
            arrivals, routing_randoms, processed
        )
        
        buildup[r] = sumBuildup / (T * numQueues)

    avgBuildup[m] = np.mean(buildup)
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)

print("Simulation complete! Generating plot...")

# Generate Figure showing buildup with fixed routing
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='green')
ax2.axvline(x=1/3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
ax2.set_title('Fixed Routing (Queue 0→Server 0, Queue 1→50/50 Split)')
plt.show()

print(ratioArr)
print(avgBuildup)