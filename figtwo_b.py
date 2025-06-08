import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

all_base_params = []
sample = 200
numQueues = 5
numServers = 6

T = 50000
rate = 1/math.sqrt(T)
gamma = rate
stepsize = 0.02
ratio = 0.0

M = 41

#generate base parameters
for i in range(sample):
    mus = np.random.uniform(0, 1, numServers)
    lambdas = np.random.uniform(0, 1, numQueues)
    all_base_params.append({'mus': mus, 'lambdas': lambdas})

##################
###WITH BUFFERS###
#################
avgBuildup = np.empty(M)
buildup97_5 = np.empty(M)
buildup2_5 = np.empty(M)
ratioArr = np.empty(M)
for m in range(M):
    if (ratio >= 0.02 and ratio < 0.12) or ratio >= 0.86:
       ratio += 0.1
       if ratio > 1:
            ratio = 1
    else:
        ratio += stepsize
    buildup = np.empty(sample)
    ratioArr[m] = ratio
    bigCount = 0
    for r in range(sample):
        processRates = all_base_params[r]['mus']
        inputRates_raw = all_base_params[r]['lambdas']

        #rescale
        lambda_sum = np.sum(inputRates_raw)
        mu_sum = np.sum(processRates)
        scaling_factor = (ratio * mu_sum) / lambda_sum
        inputRates = inputRates_raw * scaling_factor
        for i in inputRates:
            if i > 1:
                maxL = max(inputRates)
                inputRates /= maxL
                processRates /= maxL
                break

        #initialize 
        weights = np.full((numQueues, numServers), 1/numServers)
        costs = np.zeros((numQueues, numServers))
        counter = np.zeros(numServers)
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

                #update weights
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
        for q in queues:
            if q > math.sqrt(T):
                bigCount += 1
                break
        sumBuildup = sum(queues)
        buildup[r] = sumBuildup/(T*numQueues)
    avgBU = sum(buildup)/len(buildup)
    avgBuildup[m] = avgBU
    buildup97_5[m] = np.percentile(buildup, 97.5)
    buildup2_5[m] = np.percentile(buildup, 2.5)
    bigArr = np.concatenate((np.full(bigCount, 1), np.full(sample-bigCount, 0)))
    


################
###NO BUFFERS###
################
avgBuildupnb = np.empty(M)
buildup95nb = np.empty(M)
buildup5nb = np.empty(M)
ratio = 0.0
for m in range(M):
    if (ratio >= 0.02 and ratio < 0.12) or ratio >= 0.86:
        ratio += 0.1
        if ratio > 1:
            ratio = 1
    else:
        ratio += stepsize
    buildupnb = np.empty(sample)
    bigCountnb = 0
    for r in range(sample):
        processRates = all_base_params[r]['mus']
        inputRates_raw = all_base_params[r]['lambdas']

        #rescale
        lambda_sum = np.sum(inputRates_raw)
        mu_sum = np.sum(processRates)
        scaling_factor = (ratio * mu_sum) / lambda_sum
        inputRates = inputRates_raw * scaling_factor
        for i in inputRates:
            if i > 1:
                maxL = max(inputRates)
                inputRates /= maxL
                processRates /= maxL
                break

        #initialize
        weights = np.full((numQueues, numServers), 1/numServers)
        costs = np.zeros((numQueues, numServers))
        counter = np.zeros(numServers)
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
                #if processed choose packet at random 
                if(processed[k]):
                    if (len(servers[k]) > 0):
                        chosenPackets[k] = np.random.choice(servers[k])
                        costs[chosenPackets[k], k] = -1
                        counter[chosenPackets[k]] += 1

                #update weights
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
        for q in queues:
            if q > math.sqrt(T):
                bigCountnb += 1
                break
        sumBuildup = sum(queues)
        buildupnb[r] = sumBuildup/(T*numQueues)

    avgBU = sum(buildupnb)/len(buildupnb)
    avgBuildupnb[m] = avgBU
    buildup95nb[m] = np.percentile(buildupnb, 95)
    buildup5nb[m] = np.percentile(buildupnb, 5)
    bigArr = np.concatenate((np.full(bigCountnb, 1), np.full(sample-bigCountnb, 0)))


#Generates Figure 2B showing buildup vs arrival to capacity ratio in randomized simulation
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup, color='blue')
ax2.fill_between(ratioArr, buildup97_5, buildup2_5, alpha = 0.1, color='blue')
ax2.axvline(x = 1/3, color = 'red', linestyle='--')
ax2.axvline(x = 0.5, color = 'green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Buildup', fontsize=14)
plt.show()
