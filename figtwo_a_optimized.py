import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit

T = 50000
rate = 1/math.sqrt(T)
gamma = rate

sample = 200
stepsize = 0.01
numQueues = 3
numServers = 3
M = 31

inputRates = np.array([0.0, 0.0, 0.0])
processRates = np.array([0.8, 0.2, 0.2])

@jit(nopython=True)
def sample_from_weights(weights, random_val):
    """Fast weighted sampling using pre-generated random number"""
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, random_val * cumsum[-1])

@jit(nopython=True)
def run_single_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                         arrivals, noise_choices, random_servers, processed, 
                         weight_randoms):
    """Optimized simulation loop using Numba JIT compilation"""
    
    # Initialize state
    weights = np.full((numQueues, numServers), 1.0/numServers)
    queues = np.zeros(numQueues)
    buffers = np.full(numServers, -1, dtype=np.int32)
    
    gamma_over_servers = gamma / numServers
    one_minus_gamma = 1.0 - gamma
    
    for i in range(T):
        # Arrivals
        queues += arrivals[:, i]
        
        # Server selection for active queues
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        costs = np.zeros((numQueues, numServers))
        
        for j in range(numQueues):
            if queues[j] > 0:
                if noise_choices[j, i]:
                    chosen_servers[j] = random_servers[j, i]
                else:
                    chosen_servers[j] = sample_from_weights(weights[j], weight_randoms[j, i])
        
        # Process each server
        for k in range(numServers):
            # Find which queues sent packets here
            sending_queues = []
            for j in range(numQueues):
                if chosen_servers[j] == k:
                    sending_queues.append(j)
            
            # Handle buffer logic
            chosen_queue = -1
            if len(sending_queues) > 0:
                if buffers[k] == -1:  # Buffer empty
                    # Randomly accept one packet
                    idx = np.random.randint(len(sending_queues))
                    chosen_queue = sending_queues[idx]
                    buffers[k] = chosen_queue
                    costs[chosen_queue, k] = -1.0
            
            # Process buffer if packet present
            if buffers[k] != -1 and processed[k, i]:
                buffers[k] = -1
            
            # Update weights and remove packets
            for j in sending_queues:
                # Update weights
                cost = costs[j, k] / (gamma_over_servers + one_minus_gamma * weights[j, k])
                weights[j, k] *= np.exp(gamma_over_servers * (-cost))
                
                # Remove packet if it was accepted
                if j == chosen_queue:
                    queues[j] -= 1
            
            # Normalize weights for queues that sent packets
            for j in sending_queues:
                weight_sum = np.sum(weights[j])
                weights[j] /= weight_sum
    
    return np.sum(queues)

##################
###WITH BUFFERS###
##################
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)

print("Starting optimized simulation...")

for m in range(M):
    if inputRates[0] < 0.28:
        inputRates += 0.04
    else: 
        inputRates += 0.005
    
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates)/sum(processRates)
    
    # Pre-generate ALL random numbers for this rate ratio
    arrivals_batch = np.random.binomial(1, inputRates[:, np.newaxis, np.newaxis], 
                                       size=(numQueues, sample, T))
    noise_choices_batch = np.random.binomial(1, gamma, size=(numQueues, sample, T))
    random_servers_batch = np.random.randint(0, numServers, size=(numQueues, sample, T))
    processed_batch = np.random.binomial(1, processRates[:, np.newaxis, np.newaxis], 
                                        size=(numServers, sample, T))
    weight_randoms_batch = np.random.random(size=(numQueues, sample, T))
    
    for r in range(sample):
        # Extract pre-generated randoms for this sample
        arrivals = arrivals_batch[:, r, :]
        noise_choices = noise_choices_batch[:, r, :]
        random_servers = random_servers_batch[:, r, :]
        processed = processed_batch[:, r, :]
        weight_randoms = weight_randoms_batch[:, r, :]
        
        sumBuildup = run_single_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            arrivals, noise_choices, random_servers, processed, weight_randoms
        )
        
        buildup[r] = sumBuildup / (T * numQueues)

    avgBU = np.mean(buildup)
    avgBuildup[m] = avgBU
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)
    
    print(f"Completed {m+1}/{M} rate ratios (ratio: {ratioArr[m]:.3f})")

print("Simulation complete! Generating plot...")

# Generate Figure 2A
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='blue')
ax2.axvline(x=1/3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
plt.show()