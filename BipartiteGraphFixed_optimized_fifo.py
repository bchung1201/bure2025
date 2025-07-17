import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit
from numba.typed import List

T = 150000
sample = 200
stepsize = 0.01
numQueues = 3
numServers = 3
M = 40


inputRates = np.array([0.0, 0.0, 0.0])
processRates = np.array([0.8, 0.2, 0.2])

# Define accessible servers for each queue
accessibleServers = [[0,1,2], [0,1,2], [0,1,2]]

# numba doesn't directly support 2D lists, have to use special List() and copy over
numba_accessibleServers = List()
for lst in accessibleServers:
    numba_accessibleServers.append(lst)


@jit(nopython=True)
def run_fixed_routing_simulation(inputRates, processRates, T, numQueues, numServers, numba_accessibleServers):
    """Fixed routing simulation with timestamp-based FIFO"""

    # Instead of just counting, we need to track timestamps
    # Using lists to store timestamps for each queue
    queue_timestamps = List()
    for i in range(numQueues):
        queue_timestamps.append(List.empty_list(np.int64))

    # Buffers now store (queue_id, timestamp) as a tuple
    # -1 means empty, otherwise stores queue_id
    buffer_queue_ids = np.full(numServers, -1, dtype=np.int32)
    buffer_timestamps = np.full(numServers, -1, dtype=np.int64)

    for t in range(T):
        # Arrivals - add new packets with current timestamp
        for i in range(numQueues):
            num_arrivals = np.random.binomial(1, inputRates[i])
            for _ in range(num_arrivals):
                queue_timestamps[i].append(t)

        # Server selection - each queue chooses a server for its oldest packet
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        for i in range(numQueues):
            if len(queue_timestamps[i]) > 0:
                chosen_servers[i] = numba_accessibleServers[i][np.random.randint(len(numba_accessibleServers[i]))]

        # Process each server
        processed = np.zeros(numServers, dtype=np.int32)
        for i in range(numServers):
            processed[i] = np.random.binomial(1, processRates[i])

        for k in range(numServers):
            # Find which queues sent packets to this server
            sending_queues = List.empty_list(np.int32)
            sending_timestamps = List.empty_list(np.int64)

            for q in range(numQueues):
                if chosen_servers[q] == k and len(queue_timestamps[q]) > 0:
                    sending_queues.append(q)
                    # Get the timestamp of the oldest packet in this queue
                    sending_timestamps.append(queue_timestamps[q][0])

            # Handle buffer logic
            if len(sending_queues) > 0:
                if buffer_queue_ids[k] == -1:  # Buffer empty
                    # Find the queue with the oldest packet
                    oldest_idx = 0
                    oldest_time = sending_timestamps[0]
                    for idx in range(1, len(sending_timestamps)):
                        if sending_timestamps[idx] < oldest_time:
                            oldest_time = sending_timestamps[idx]
                            oldest_idx = idx

                    chosen_queue = sending_queues[oldest_idx]
                    # Move packet to buffer
                    buffer_queue_ids[k] = chosen_queue
                    buffer_timestamps[k] = queue_timestamps[chosen_queue][0]
                    # Remove from queue
                    queue_timestamps[chosen_queue].pop(0)

            # Process buffer and remove packet only if successful
            if buffer_queue_ids[k] != -1 and processed[k] == 1:
                buffer_queue_ids[k] = -1
                buffer_timestamps[k] = -1

    # Calculate total queue buildup
    total_buildup = 0
    for i in range(numQueues):
        total_buildup += len(queue_timestamps[i])

    return total_buildup


##################
###FIXED ROUTING###
##################
print("Starting fixed routing simulation with timestamp-based FIFO...")
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)

for m in range(M):
    print(f"Reached m: {m}")
    inputRates += 0.01
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)
    for r in range(sample):
        sumBuildup = run_fixed_routing_simulation(inputRates, processRates, T, numQueues, numServers,
                                                  numba_accessibleServers)
        buildup[r] = sumBuildup / (T * numQueues)

    avgBuildup[m] = np.mean(buildup)
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)

print("Simulation complete! Generating plot...")

# Generate Figure showing buildup with fixed routing
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='green')
ax2.axvline(x=1 / 3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
ax2.set_title('Fixed Routing with FIFO (Timestamp-based)')
plt.show()

print(ratioArr)
print(avgBuildup)