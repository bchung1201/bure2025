import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import deque

T = 50000
rate = 1 / math.sqrt(T)
gamma = rate

# sample = 200
sample = 200
stepsize = 0.01
numQueues = 2
numServers = 2
M = 3

inputRates = np.array([0.37, 0.37])
processRates = np.array([0.6, 0.2])
accessibleServers = [[0], [0, 1]]

class Packet:
    def __init__(self, timestamp, queue_id):
        self.timestamp = timestamp
        self.queue_id = queue_id

##################
###WITH BUFFERS###
##################
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)
for m in range(M):
    print("Reached m: " + str(m))
    '''if inputRates[0] < 0.28:
        inputRates += 0.03
    else:
        inputRates += 0.03'''
    inputRates += 0.01
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)
    for r in range(sample):
        # initialize
        weights = [np.full(len(accessibleServers[i]), 1/len(accessibleServers[i])) for i in range(numQueues)]
        costs = [np.zeros(len(accessibleServers[i])) for i in range(numQueues)]
        chosenServers = np.ones(numQueues) * -1
        queues = [deque() for i in range(numQueues)]  # Changed to deque for FIFO
        servers = [[] for i in range(numServers)]
        buffers = [None for i in range(numServers)]  # Store actual packet objects
        counter = np.zeros(numQueues)

        for i in range(T):
            chosenServers = np.ones(numQueues) * -1
            costs = [np.zeros(len(accessibleServers[i])) for i in range(numQueues)]
            servers = [[] for i in range(numServers)]
            # packets arrive based on inputRate to each queue
            for j in range(numQueues):
                input = np.random.binomial(1, inputRates[j])
                if (input == 1):
                    # Create packet with timestamp and add to queue
                    packet = Packet(i, j)
                    queues[j].append(packet)

            # each queue sends packet to a chosen server
            for j in range(numQueues):
                if (len(queues[j]) > 0):
                    noise = np.random.binomial(1, gamma)
                    # with prob gamma choose server at random
                    if (noise):
                        choice = np.random.choice(accessibleServers[j])
                    # with prob 1-gamma choose server based on weights
                    else:
                        choice = np.random.choice(accessibleServers[j], p=weights[j])

                    # Send oldest packet (FIFO)
                    oldest_packet = queues[j].popleft()
                    servers[choice].append(oldest_packet)
                    chosenServers[j] = choice

            # with prob processRates[i] then server i able to clear packet in buffer
            processed = np.random.binomial(1, processRates)
            chosenPackets = [None for i in range(numServers)]
            for k in range(numServers):
                # if buffer empty then choose arriving packet to place in there
                if (buffers[k] is None):
                    if (len(servers[k]) > 0):
                        # Choose oldest packet among all packets sent to this server
                        oldest_packet = min(servers[k], key=lambda p: p.timestamp)
                        chosenPackets[k] = oldest_packet
                        buffers[k] = oldest_packet
                        queue_id = oldest_packet.queue_id
                        for j in range(len(costs[queue_id])):
                            if accessibleServers[queue_id][j] == k:
                                costs[queue_id][j] = -1
                                #updates weights
                                queue = queue_id
                                cost = costs[queue][j] / ((gamma / len(accessibleServers[queue])) + ((1 - gamma) * weights[queue][j]))
                                weights[queue][j] *= np.exp((gamma / len(accessibleServers[queue])) * (-1) * cost)
                        counter[queue_id] += 1
                # if buffer full and processed then clear buffer
                if (buffers[k] is not None and processed[k] == 1):
                    buffers[k] = None

            for j in range(numQueues):
                # rescale weights
                weights[j] = weights[j] / sum(weights[j])
                # Check if packet was processed - no need to remove from queue as it was already removed

        # Calculate buildup based on queue lengths
        total_packets = sum(len(queue) for queue in queues)
        buildup[r] = total_packets / (T * numQueues)

    avgBU = sum(buildup) / len(buildup)
    avgBuildup[m] = avgBU
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)

# Generates Figure 2A showing buildup in a symmetric system
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='blue')
ax2.axvline(x=1 / 3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
plt.show()