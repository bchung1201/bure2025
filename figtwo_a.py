import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

T = 50000
rate = 1/math.sqrt(T)
gamma = rate

sample = 200
stepsize = 0.01
numQueues = 3
numServers = 3
M = 31

inputRates = np.array([0.0, 0.0, 0.0])
inputRatesnb = np.array([0.0, 0.0, 0.0])
processRates = np.array([0.8, 0.2, 0.2])

##################
###WITH BUFFERS###
##################
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)
for m in range(M):
    if inputRates[0] < 0.28:
        inputRates += 0.04
    else: 
        inputRates += 0.005
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates)/sum(processRates)
    for r in range(sample):
        #initialize 
        weights = np.full((numQueues, numServers), 1/numServers)
        costs = np.zeros((numQueues, numServers))
        counter = np.zeros(numQueues)
        chosenServers = np.ones(numQueues) * -1
        queues = np.zeros(numQueues)
        servers = [[] for i in range(numServers)]
        buffers = np.ones(numServers) * -1

        for i in range(T):
            chosenServers = np.ones(numQueues) * -1
            costs = np.zeros((numQueues, numServers))
            servers = [[] for i in range(numServers)]
            #packets arrive based on inputRate to each queue
            for j in range(numQueues):
                input = np.random.binomial(1, inputRates[j])
                if(input == 1):
                    queues[j] += 1

            #each queue sends packet to a choosen server
            for j in range(numQueues):
                if(queues[j] > 0):
                    noise = np.random.binomial(1, gamma)
                    #with prob gamma choose server at random
                    if (noise):
                        choice = np.random.choice(numServers)
                    #with prob 1-gamma choose server based on weights
                    else:
                        choice = np.random.choice(numServers, p=weights[j])

                    servers[choice].append(j)
                    chosenServers[j] = choice
            
            #with prob processRates[i] then server i able to clear packet in buffer
            processed = np.random.binomial(1, processRates)
            chosenPackets = np.ones(numServers, dtype=int) * -1
            for k in range(numServers):
                #if buffer empty then choose arriving packet to place in there
                if(buffers[k] == -1):
                    if (len(servers[k]) > 0):
                        #randomly choose from queues that sent packet here
                        chosenPackets[k] = np.random.choice(servers[k])
                        buffers[k] = chosenPackets[k]
                        costs[chosenPackets[k], k] = -1
                        counter[chosenPackets[k]] += 1
                #if buffer full and processed then clear buffer
                if(buffers[k] != -1 and processed[k] == 1):     
                    buffers[k] = -1

                #updates weights
                for j in servers[k]:
                    cost = costs[j, k]/((gamma/numServers)+((1-gamma)*weights[j, k]))
                    weights[j, k] *= np.exp((gamma/numServers)*(-1)*cost)
            
            for j in range(numQueues):
                #rescale weights
                weights[j] = weights[j]/sum(weights[j])
                #if packet was cleared and remove from queue
                for k in range(numServers):
                    if(chosenServers[j] == k and chosenPackets[k] == j):
                        queues[j] -= 1

        sumBuildup = sum(queues)
        buildup[r] = sumBuildup/(T*numQueues)

    avgBU = sum(buildup)/len(buildup)
    avgBuildup[m] = avgBU
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)
    
#Generates Figure 2A showing buildup in a symmetric system
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha = 0.1, color='blue')
ax2.axvline(x = 1/3, color = 'red', linestyle='--')
ax2.axvline(x = 0.5, color = 'green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
plt.show()

