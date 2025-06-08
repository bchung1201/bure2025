import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


inputRates = [0.6, 0.3, 0.3, 0.3]
processRates = [0.8, 0.4, 0.4, 0.2, 0.2]

numQueues = len(inputRates)
numServers = len(processRates)
T = 10000
rate = 1/math.sqrt(T)
gamma = rate
sample = 1000

##################
###WITH BUFFERS###
##################
serviceRates = []
serviceRatesQueues = [[] for i in range(numQueues)]
serviceRatesServers = [[] for i in range(numServers)]

for r in range(sample):
    #initialize 
    weights = np.array([[(1/numServers) for i in range(numServers)] for j in range(numQueues)])
    for j in range(numQueues):
        weights[j] = weights[j]/sum(weights[j])

    costs = [[0 for i in range(numServers)] for j in range(numQueues)]
    counterQ = [0 for i in range(numQueues)]
    counterS = [0 for i in range(numServers)]
    chosenServers = [-1 for i in range(numQueues)]
    queues = [[] for i in range(numQueues)]
    servers = [[] for i in range(numServers)]
    buffers = [-1 for i in range(numServers)]


    for i in range(T):
        chosenServers = [-1 for i in range(numQueues)]
        costs = [[0 for i in range(numServers)] for j in range(numQueues)]
        servers = [[] for i in range(numServers)]
        #packets arrive based on inputRate to each queue
        for j in range(numQueues):
            input = np.random.binomial(1, inputRates[j])
            if(input == 1):
                queues[j].append(i)

        #each queue sends packet to a choosen server
        for j in range(numQueues):
            if(len(queues[j]) > 0):
                if(len(queues[j]) > 0):
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
        chosenPackets = [-1 for i in range(numServers)]
        for k in range(numServers):
            #if buffer empty then choose arriving packet to place in there
            if(buffers[k] == -1):
                if (len(servers[k]) > 0):
                    #randomly choose from queues that sent packet here
                    chosenPackets[k] = np.random.choice(servers[k])
                    buffers[k] = chosenPackets[k]
                    costs[chosenPackets[k]][k] = -1
                    counterQ[chosenPackets[k]] += 1
            #if buffer full and processed then clear buffer
            if(buffers[k] != -1 and processed[k] == 1):     
                counterS[k] += 1
                buffers[k] = -1
            
            #update weights     
            for j in servers[k]:
                cost = costs[j][k]/((gamma/numServers)+((1-gamma)*weights[j][k]))
                weights[j][k] *= np.exp(((gamma/numServers)*(-1)*cost))


        for j in range(numQueues):
            #rescale weights
            weights[j] = weights[j]/sum(weights[j])
            #if packet was cleared and remove from queue
            for k in range(numServers):
                if(chosenServers[j] == k and chosenPackets[k] == j):
                    queues[j].pop(0)
            

    serviceRates.append(sum(counterQ)/T)
    for j in range(numQueues):    
        serviceRatesQueues[j].append(counterQ[j]/T)
    for k in range(numServers):
        serviceRatesServers[k].append(counterS[k]/T)

################
###NO BUFFERS###
################
serviceRatesnb = []
serviceRatesQueuesnb = [[] for i in range(numQueues)]
serviceRatesServersnb = [[] for i in range(numServers)]
for r in range(sample):
    
    #initialize 
    weights = np.array([[(1/numServers) for i in range(numServers)] for j in range(numQueues)])
    for j in range(numQueues):
        weights[j] = weights[j]/sum(weights[j])

    costs = [[0 for i in range(numServers)] for j in range(numQueues)]
    counterQ = [0 for i in range(numQueues)]
    counterS = [0 for i in range(numServers)]
    chosenServers = [-1 for i in range(numQueues)]
    queues = [[] for i in range(numQueues)]
    servers = [[] for i in range(numServers)]


    for i in range(T):
        chosenServers = [-1 for i in range(numQueues)]
        costs = [[0 for i in range(numServers)] for j in range(numQueues)]
        servers = [[] for i in range(numServers)]
        #packets arrive based on inputRate to each queue
        for j in range(numQueues):
            input = np.random.binomial(1, inputRates[j])
            if(input == 1):
                queues[j].append(i)

        #each queue sends packet to a choosen server
        for j in range(numQueues):
            if(len(queues[j]) > 0):
                if(len(queues[j]) > 0):
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
        chosenPackets = [-1 for i in range(numServers)]
        for k in range(numServers):
            #if processed choose packet at random 
            if(processed[k]):
                if (len(servers[k]) > 0):
                    chosenPackets[k] = np.random.choice(servers[k])
                    costs[chosenPackets[k]][k] = -1
                    counterS[k] += 1
                    counterQ[chosenPackets[k]] += 1
            
            #update weights
            for j in servers[k]:
                cost = costs[j][k]/((gamma/numServers)/+((1-gamma)*weights[j][k]))
                weights[j][k] *= np.exp(((gamma/numServers)*(-1)*cost))


        for j in range(numQueues):
            #rescale weights
            weights[j] = weights[j]/sum(weights[j])
            #if packet was cleared and remove from queue
            for k in range(numServers):
                if(chosenServers[j] == k and chosenPackets[k] == j):
                    queues[j].pop(0)
            

    serviceRatesnb.append(sum(counterQ)/T)
    for k in range(numServers):
        serviceRatesServersnb[k].append(counterS[k]/T)

#this plots figure 3B the clearing rates of buffers vs no buffers
fig, ax1 = plt.subplots()
sns.histplot(data=[serviceRates, serviceRatesnb], bins=50, kde=True, palette=['blue', 'orange'])
ax1.set_xlabel('Clearing Rates', fontsize=14)
ax1.set_ylabel('Count', fontsize=14)
legend = ax1.get_legend()
handles = legend.legend_handles
ax1.legend(handles, ['Buffers', 'No Buffers'], fontsize=14)
plt.show()

