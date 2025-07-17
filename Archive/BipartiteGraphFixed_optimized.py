import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit
from numba.typed import List

T = 150000
sample = 200
stepsize = 0.01
numQueues = 1
numServers = 2
M = 1

inputRates = np.array([0.55])
processRates = np.array([0.3, 0.3])

# Define accessible servers for each queue
accessibleServers = [[0, 1]]
#numba doesn't directly support 2D lists, have to use special List() and copy over
numba_accessibleServers = List()
for lst in accessibleServers:
    numba_accessibleServers.append(lst)

@jit(nopython=True)
#passing in all the global variables again as locals is required for jit to work properly
def run_fixed_routing_simulation(inputRates, processRates, T, numQueues, numServers, numba_accessibleServers):
    """Fixed routing simulation - no learning, just fixed probabilities"""
    
    queues = np.zeros(numQueues)
    buffers = np.full(numServers, -1, dtype=np.int32)
    
    for t in range(T):
        # Arrivals
        for i in range(numQueues):
            queues[i] += np.random.binomial(1, inputRates[i])

        #server selection
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        for i in range(numQueues):
            if queues[i] > 0:
                chosen_servers[i] = numba_accessibleServers[i][np.random.randint(len(numba_accessibleServers[i]))]

        processed = [-1 for i in range(numServers)]

        for i in range(numServers):
            processed[i] = np.random.binomial(1, processRates[i])

        for k in range(numServers):
            # Find which queues sent packets to this server
            sending_queues = []
            for q in range(numQueues):
                if chosen_servers[q] == k:
                    sending_queues.append(q)
            
            # Handle buffer logic
            if len(sending_queues) > 0:
                if buffers[k] == -1:  # Buffer empty
                    # Randomly accept one packet
                    idx = np.random.randint(len(sending_queues))
                    chosen_queue = sending_queues[idx]
                    buffers[k] = chosen_queue
                    queues[chosen_queue] -= 1 #Remove from queue

                # Accept packet into buffer

            # Process buffer and remove packet only if successful
            if buffers[k] != -1 and processed[k] == 1:
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
    inputRates += 0
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)
    for r in range(sample):
        sumBuildup = run_fixed_routing_simulation(inputRates, processRates, T, numQueues, numServers, numba_accessibleServers)
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