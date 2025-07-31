import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit
from numba.typed import List

T = 25000
rate = 1 / math.sqrt(T)
gamma = rate

useTimeStamps = True

sample = 10
numQueues = 350
numServers = 350

M = 24

#0.01 means all input rates increase by 1% each m in M (additive, not multiplicative)
inputRateStepSize = 0.4

lst = [2/((numQueues-1)**(1/3))]
for i in range(numQueues-1):
    lst.append(1/((numQueues-1)**(1/3)))
inputRates = np.array(lst) / 3

lst2 = [1.0]
for i in range(numQueues-1):
    lst2.append(0.5)
processRates = np.array(lst2)

# Define accessible servers for each queue
lst3 = [[0]]
for i in range(numQueues-1):
    lst3.append([0, i+1])
accessibleServers = lst3

startingQueues = np.array([0 for i in range(numQueues)])

########################################################
###AFTER HERE THERE ARE NO MORE CHANGEABLE PARAMETERS###
########################################################

inputRateStep = np.array([inputRateStepSize*inputRates[i] for i in range(numQueues)])

# Create typed list for numba - need to specify type since lists are empty
from numba import types

queues = List()
for i in range(numQueues):
    queues.append(List.empty_list(types.int32))

# Pre-compute mapping for optimization
max_accessible = max(len(servers) for servers in accessibleServers)
accessible_matrix = np.full((numQueues, max_accessible), -1, dtype=np.int32)
accessible_lengths = np.zeros(numQueues, dtype=np.int32)

for q in range(numQueues):
    accessible_lengths[q] = len(accessibleServers[q])
    for i, server in enumerate(accessibleServers[q]):
        accessible_matrix[q, i] = server


@jit(nopython=True)
def sample_from_weights(weights, random_val):
    """Fast weighted sampling using pre-generated random number"""
    cumsum = np.cumsum(weights)
    if cumsum[-1] == 0:
        return np.random.randint(len(weights))
    return np.searchsorted(cumsum, random_val * cumsum[-1])

@jit(nopython=True)
def run_bipartite_simulation(inputRates, processRates, T, gamma, numQueues, numServers,
                            accessible_matrix, accessible_lengths, queues, useTimeStamps):
    """Optimized bipartite simulation with JIT compilation"""

    # Initialize weights - different sizes for each queue
    weights = np.zeros((numQueues, max_accessible))
    for q in range(numQueues):
        for i in range(accessible_lengths[q]):
            weights[q, i] = 1.0 / accessible_lengths[q]

    buffers = np.full(numServers, -1, dtype=np.int32)

    # Track actual send rates: count how many times each queue sends to each server
    # send_counts = np.zeros((numQueues, max_accessible), dtype=np.int32)
    # total_sends = np.zeros(numQueues, dtype=np.int32)

    # Reset queues at start of each simulation
    for q in range(numQueues):
        queues[q].clear()
        # Add starting packets to each queue
        for _ in range(startingQueues[q]):
            if useTimeStamps:
                queues[q].append(0)  # All starting packets have timestamp 0
            else:
                queues[q].append(1)

    for t in range(T):
        # Arrivals
        for q in range(numQueues):
            rng = np.random.binomial(1, inputRates[q])
            if rng == 1:
                if useTimeStamps:
                    queues[q].append(t)
                else:
                    queues[q].append(1)

        # Server selection for active queues
        chosen_servers = np.full(numQueues, -1, dtype=np.int32)
        costs = np.zeros((numQueues, max_accessible))

        for q in range(numQueues):
            if len(queues[q]) > 0:
                accessible_count = accessible_lengths[q]

                if np.random.binomial(1, gamma) == 1:
                    # Random choice from accessible servers
                    server_idx = int(np.random.random() * accessible_count)
                    chosen_servers[q] = accessible_matrix[q, server_idx]
                else:
                    # Weighted choice from accessible servers
                    active_weights = weights[q, :accessible_count]
                    server_idx = sample_from_weights(active_weights, np.random.random())
                    chosen_servers[q] = accessible_matrix[q, server_idx]

                # Track that this queue sent to a server
                # send_counts[q, server_idx] += 1
                # total_sends[q] += 1

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
                    oldest_queues = []
                    min_time = T+1
                    for q in sending_queues:
                        min_time = min(min_time, queues[q][0])
                    for q in sending_queues:
                        if queues[q][0] == min_time:
                            oldest_queues.append(q)
                    idx = np.random.randint(len(oldest_queues))
                    chosen_queue = oldest_queues[idx]
                    buffers[k] = chosen_queue

                    # FIXED: Correct weight update logic
                    # Find which index in the accessible servers corresponds to server k
                    for j in range(accessible_lengths[chosen_queue]):
                        if accessible_matrix[chosen_queue, j] == k:
                            costs[chosen_queue, j] = -1.0
                            # Update weights
                            cost = costs[chosen_queue, j] / ((gamma / accessible_lengths[chosen_queue]) +
                                                           ((1 - gamma) * weights[chosen_queue, j]))
                            weights[chosen_queue, j] *= np.exp((gamma / accessible_lengths[chosen_queue]) * (-cost))
                            break

            # Process buffer if packet present
            if buffers[k] != -1 and np.random.binomial(1, processRates[k]) == 1:
                buffers[k] = -1

            # Remove packet if it was accepted
            if chosen_queue != -1:
                queues[chosen_queue].pop(0)

        # Normalize weights for all queues
        for q in range(numQueues):
            weight_sum = 0.0
            for j in range(accessible_lengths[q]):
                weight_sum += weights[q, j]
            if weight_sum > 0:
                for j in range(accessible_lengths[q]):
                    weights[q, j] /= weight_sum

    # Compute actual send rates (percentage of times each queue sent to each server)
    # actual_weights = np.zeros((numQueues, max_accessible))
    # for q in range(numQueues):
    #     if total_sends[q] > 0:
    #         for i in range(accessible_lengths[q]):
    #             actual_weights[q, i] = send_counts[q, i] / total_sends[q]
    #     else:
    #         # If no sends occurred, set equal weights for accessible servers
    #         for i in range(accessible_lengths[q]):
    #             actual_weights[q, i] = 1.0 / accessible_lengths[q]

    res = 0
    for q in range(numQueues):
        res += len(queues[q])
    return (res)
            #, actual_weights)


##################
###WITH BUFFERS###
##################
print("Starting optimized bipartite simulation...")
avgBuildup = np.empty(M)
buildup95 = np.empty(M)
buildup5 = np.empty(M)
ratioArr = np.empty(M)

# Initialize weights accumulator for averaging across all trials
# weights_accumulator = np.zeros((numQueues, max_accessible))
# total_trials = 0

for m in range(M):
    print(f"Reached m: {m}")
    if m > 0:
        inputRates += inputRateStep
    buildup = np.empty(sample)
    ratioArr[m] = sum(inputRates) / sum(processRates)

    for r in range(sample):
        # Extract pre-generated randoms for this sample
        sumBuildup = run_bipartite_simulation(
            inputRates, processRates, T, gamma, numQueues, numServers,
            accessible_matrix, accessible_lengths, queues, useTimeStamps
        )

        buildup[r] = sumBuildup / (T * numQueues)

        # Accumulate weights for averaging
        # weights_accumulator += weights
        # total_trials += 1

    avgBuildup[m] = np.mean(buildup)
    buildup95[m] = np.percentile(buildup, 95)
    buildup5[m] = np.percentile(buildup, 5)

print("Simulation complete! Generating plot...")

# Generate Figure showing buildup in a bipartite system
fig2, ax2 = plt.subplots()
ax2.plot(ratioArr, avgBuildup)
ax2.fill_between(ratioArr, buildup95, buildup5, alpha=0.1, color='blue')
ax2.axvline(x=1/3, color='red', linestyle='--')
ax2.axvline(x=0.5, color='green', linestyle=':')
ax2.set_xlabel('Arrival to capacity ratio', fontsize=14)
ax2.set_ylabel('Build Up', fontsize=14)
plt.title('Bipartite Graph EXP3 - Optimized')
plt.show()

print("ratioArr: ", ratioArr)
print("avgBuildup", avgBuildup)

# Calculate and print average weights across all trials
# if total_trials > 0:
#     avg_weights = weights_accumulator / total_trials
#     print(f"\nAverage weights across {total_trials} trials:")
#     print("=" * 50)
#
#     for q in range(numQueues):
#         print(f"Queue {q} -> Server weights:")
#         for i in range(accessible_lengths[q]):
#             server_id = accessible_matrix[q, i]
#             weight_val = avg_weights[q, i]
#             print(f"  Queue {q} -> Server {server_id}: {weight_val:.6f}")
#         print()
# else:
#     print("No trials completed!")