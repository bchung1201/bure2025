import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from numba import jit

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

print("Generating base parameters...")
# Generate base parameters
for i in range(sample):
    mus = np.random.uniform(0, 1, numServers)
    lambdas = np.random.uniform(0, 1, numQueues)
    all_base_params.append({'mus': mus, 'lambdas': lambdas})

@jit(nopython=True)
def sample_from_weights(weights, random_val):
    """Fast weighted sampling using pre-generated random number"""
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, random_val * cumsum[-1])

@jit(nopython=True)
def run_buffered_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                           arrivals, noise_choices, random_servers, processed, 
                           weight_randoms):
    """Optimized buffered simulation using Numba JIT"""
    
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

@jit(nopython=True)
def run_unbuffered_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                             arrivals, noise_choices, random_servers, processed, 
                             weight_randoms):
    """Optimized unbuffered simulation using Numba JIT"""
    
    weights = np.full((numQueues, numServers), 1.0/numServers)
    queues = np.zeros(numQueues)
    
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
        
        # Process each server (no buffers)
        for k in range(numServers):
            # Find which queues sent packets here
            sending_queues = []
            for j in range(numQueues):
                if chosen_servers[j] == k:
                    sending_queues.append(j)
            
            # Handle direct processing (no buffer)
            chosen_queue = -1
            if len(sending_queues) > 0 and processed[k, i]:
                # Randomly choose one packet to process
                idx = np.random.randint(len(sending_queues))
                chosen_queue = sending_queues[idx]
                costs[chosen_queue, k] = -1.0
            
            # Update weights and remove packets
            for j in sending_queues:
                # Update weights
                cost = costs[j, k] / (gamma_over_servers + one_minus_gamma * weights[j, k])
                weights[j, k] *= np.exp(gamma_over_servers * (-cost))
                
                # Remove packet if it was processed
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
print("Starting WITH BUFFERS simulation...")
avgBuildup = np.empty(M)
buildup97_5 = np.empty(M)
buildup2_5 = np.empty(M)
ratioArr = np.empty(M)

ratio = 0.0
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
    
    # Pre-generate ALL random numbers for this ratio
    arrivals_batch = np.zeros((numQueues, sample, T))
    noise_choices_batch = np.random.binomial(1, gamma, size=(numQueues, sample, T))
    random_servers_batch = np.random.randint(0, numServers, size=(numQueues, sample, T))
    processed_batch = np.zeros((numServers, sample, T))
    weight_randoms_batch = np.random.random(size=(numQueues, sample, T))
    
    for r in range(sample):
        processRates = all_base_params[r]['mus']
        inputRates_raw = all_base_params[r]['lambdas']

        # Rescale
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
        
        # Generate random numbers for this specific sample
        arrivals_batch[:, r, :] = np.random.binomial(1, inputRates[:, np.newaxis], size=(numQueues, T))
        processed_batch[:, r, :] = np.random.binomial(1, processRates[:, np.newaxis], size=(numServers, T))
        
        sumBuildup = run_buffered_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            arrivals_batch[:, r, :], noise_choices_batch[:, r, :], 
            random_servers_batch[:, r, :], processed_batch[:, r, :], 
            weight_randoms_batch[:, r, :]
        )
        
        # Check if any queue is too big
        # (Note: this check is approximate since we can't easily get individual queue sizes from JIT function)
        if sumBuildup / (T * numQueues) > math.sqrt(T) / T:
            bigCount += 1
            
        buildup[r] = sumBuildup / (T * numQueues)

    avgBuildup[m] = np.mean(buildup)
    buildup97_5[m] = np.percentile(buildup, 97.5)
    buildup2_5[m] = np.percentile(buildup, 2.5)
    
    print(f"WITH BUFFERS: Completed {m+1}/{M} ratios (ratio: {ratio:.3f})")

################
###NO BUFFERS###
################
print("Starting NO BUFFERS simulation...")
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
    
    # Pre-generate ALL random numbers for this ratio
    arrivals_batch = np.zeros((numQueues, sample, T))
    noise_choices_batch = np.random.binomial(1, gamma, size=(numQueues, sample, T))
    random_servers_batch = np.random.randint(0, numServers, size=(numQueues, sample, T))
    processed_batch = np.zeros((numServers, sample, T))
    weight_randoms_batch = np.random.random(size=(numQueues, sample, T))
    
    for r in range(sample):
        processRates = all_base_params[r]['mus']
        inputRates_raw = all_base_params[r]['lambdas']

        # Rescale
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
        
        # Generate random numbers for this specific sample
        arrivals_batch[:, r, :] = np.random.binomial(1, inputRates[:, np.newaxis], size=(numQueues, T))
        processed_batch[:, r, :] = np.random.binomial(1, processRates[:, np.newaxis], size=(numServers, T))
        
        sumBuildup = run_unbuffered_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            arrivals_batch[:, r, :], noise_choices_batch[:, r, :], 
            random_servers_batch[:, r, :], processed_batch[:, r, :], 
            weight_randoms_batch[:, r, :]
        )
        
        # Check if any queue is too big (approximate)
        if sumBuildup / (T * numQueues) > math.sqrt(T) / T:
            bigCountnb += 1
            
        buildupnb[r] = sumBuildup / (T * numQueues)

    avgBuildupnb[m] = np.mean(buildupnb)
    buildup95nb[m] = np.percentile(buildupnb, 95)
    buildup5nb[m] = np.percentile(buildupnb, 5)
    
    print(f"NO BUFFERS: Completed {m+1}/{M} ratios (ratio: {ratio:.3f})")

print("Simulation complete! Generating plot...")

# Generate Figure 2B showing buildup vs arrival to capacity ratio
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup, color='blue', label='With Buffers')
ax2.fill_between(ratioArr, buildup97_5, buildup2_5, alpha=0.1, color='blue')
ax2.plot(ratioArr, avgBuildupnb, color='red', label='No Buffers')
ax2.fill_between(ratioArr, buildup95nb, buildup5nb, alpha=0.1, color='red')
ax2.axvline(x=1/3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Buildup', fontsize=14)
ax2.legend()
plt.show()