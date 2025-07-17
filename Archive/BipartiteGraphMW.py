import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

T = 5000
# sample = 200
sample = 100
M = 10

stepsize = 0.01
numQueues = 1
numServers = 1

learning_rate = math.sqrt(math.log(numServers) / T)

inputRates = np.array([0.4])
processRates = np.array([0.6])
accessibleServers = [[0]]

##################
###WITH BUFFERS###
##################
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)
for m in range(M):
    print("Reached m: " + str(m))
    # if inputRates[0] < 0.28:
    #     inputRates += 0.03
    # else:
    #     inputRates += 0.03
    inputRates += 0.02
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)
    for r in range(sample):
        # initialize
        weights = [np.full(len(accessibleServers[i]), 1/len(accessibleServers[i])) for i in range(numQueues)]
        costs = [np.zeros(len(accessibleServers[i])) for i in range(numQueues)]
        chosenServers = np.ones(numQueues) * -1
        queues = np.zeros(numQueues)
        servers = [[] for i in range(numServers)]
        buffers = np.ones(numServers) * -1
        counter = np.zeros(numQueues)

        for i in range(T):
            chosenServers = np.ones(numQueues) * -1
            costs = [np.zeros(len(accessibleServers[i])) for i in range(numQueues)]
            servers = [[] for i in range(numServers)]
            # packets arrive based on inputRate to each queue
            for j in range(numQueues):
                input = np.random.binomial(1, inputRates[j])
                if (input == 1):
                    queues[j] += 1

            # each queue sends packet to a chosen server
            for j in range(numQueues):
                if (queues[j] > 0):
                    choice = np.random.choice(accessibleServers[j], p=weights[j])
                    servers[choice].append(j)
                    chosenServers[j] = choice

            # with prob processRates[i] then server i able to clear packet in buffer
            processed = np.random.binomial(1, processRates)
            chosenPackets = np.ones(numServers, dtype=int) * -1
            for k in range(numServers):
                # if buffer empty then choose arriving packet to place in there
                if (buffers[k] == -1):
                    if (len(servers[k]) > 0):
                        # randomly choose from queues that sent packet here
                        chosenPackets[k] = np.random.choice(servers[k])
                        buffers[k] = chosenPackets[k]
                        priorities = np.random.permutation(numQueues)
                        for j in range(len(costs[chosenPackets[k]])):
                            if accessibleServers[chosenPackets[k]][j] == k:
                                costs[chosenPackets[k]][j] = -1

                        for q in range(len(priorities)):
                            for j in range(len(costs[priorities[q]])):
                                if accessibleServers[chosenPackets[k]][j] == k:
                                    costs[priorities[q]][j] = -1
                                    break
                            if priorities[q] == chosenPackets[k]:
                                break

                        counter[chosenPackets[k]] += 1
                # if buffer full and processed then clear buffer
                if (buffers[k] != -1 and processed[k] == 1):
                    buffers[k] = -1

            #updates weights
            for j in range(numQueues):
                for k in range(len(accessibleServers[j])):
                    cost = costs[j][k]
                    weights[j][k] *= 1-learning_rate*cost

            for j in range(numQueues):
                # rescale weights
                weights[j] = weights[j] / sum(weights[j])
                # if packet was cleared and remove from queue
                for k in range(numServers):
                    if (chosenServers[j] == k and chosenPackets[k] == j):
                        queues[j] -= 1

        sumBuildup = sum(queues)
        buildup[r] = sumBuildup / (T * numQueues)

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

