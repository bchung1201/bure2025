import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#return total variation distance between the distributions
def TV(P,Q):
    return np.max(abs(P - Q))

#implements the simulation for figure 4c using the EXP3 algorithm
def run_sim(sample_size, T, arrival_rates, service_rates, eta, gamma, nash, symm_nash):

    n = len(arrival_rates)
    m = len(service_rates)

    #to keep track of distance from the actual joint distribution
    emp_diff = []
    weight_diff = []


    for sample in range(sample_size):

        #joint distribution is hardcoded for n = 2
        emp = np.array([[0, 0],[0, 0]])
        some_sent = 0

        weights = np.ones((n,m))*(1/m)

        #initialize queue packets and buffers to be empty
        queue_packets = np.zeros(n)
        buffers = m*[-1]
        rng = np.random.default_rng()

        for t in range(T):
            #packets arrive and are processed according to specified rates
            arrived = np.random.binomial(size=n, n=1, p=arrival_rates)
            processed_packets = np.zeros((n,m))
            server_packets = [[] for i in range(m)]
            choices = n*[-1]

            #packet-sending logic
            for j in range(n):
                if arrived[j] > 0: queue_packets[j] += 1
                if queue_packets[j]: 
                    deviate = np.random.binomial(size=1,n=1,p=gamma)[0]
                    choice = rng.choice(m) if deviate else rng.choice(m,p=weights[j])
                    choices[j] = choice
                    server_packets[choice].append(j)

            #update joint distribution
            if choices[0] > -1 and choices[1] > -1: 
                some_sent += 1
                emp[choices[0]][choices[1]] += 1

            #packet-clearing logic
            processed = np.random.binomial(size=m,n=1,p=service_rates)
            for k in range(m):
                if server_packets[k] and buffers[k] == -1:
                    chosen_queue = rng.choice(server_packets[k])
                    buffers[k] = chosen_queue
                    queue_packets[chosen_queue] -= 1
                    processed_packets[chosen_queue][k] = -1
                    if processed[k] > 0:
                        buffers[k] = -1
                elif processed[k] and buffers[k] > -1:
                    buffers[k] = -1

            #reweighting of queues' strategies
            for j in range(n):
                if choices[j] != -1 and processed_packets[j][choices[j]] < 0:
                    w = weights[j][choices[j]]
                    cost = -1/((1-gamma)*w + gamma*(1/m))
                    weights[j][choices[j]] *= np.exp(eta*(-cost))
                    weights[j] *= 1/(1 - w + weights[j][choices[j]])


        #calculate joint distribution of sending probabilities
        joint_weights = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                w0 = weights[0][i]
                w1 = weights[1][j]
                joint_weights[i][j] = w0*w1*(1-gamma)*(1-gamma) + (1-gamma)*(gamma/2)*(w0 + w1) + (gamma/2)*(gamma/2)

        emp = emp/some_sent

        #min is the one we should actually be comparing to
        e_diff = min(TV(nash, emp), TV(symm_nash, emp))
        w_diff = min(TV(nash, joint_weights), TV(symm_nash, joint_weights))
        
        emp_diff.append(e_diff)
        weight_diff.append(w_diff)

    return np.array(emp_diff), np.array(weight_diff)

#hardcoded values for the setup in the paper

plt.rcParams.update({'font.size': 14})

horizons = [100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000, 120_000, 140_000, 150_000, 160_000, 180_000, 200_000, 250_000, 300_000, 500_000, 750_000, 1_000_000]

sample_size = 200

arrival_rates = [1/4,1/4]
service_rates = [2/3,2/3]
nash = np.array([[0,1],[0,0]])
symm_nash = np.array([[0,0],[1,0]])


weights = []

lower_weights = []

upper_weights = []

emp = []

lower_emp = []

upper_emp = []


#run simulation
for T in horizons:

    eta = 1/(np.sqrt(T))

    gamma = eta

    emp_diff, weights_diff = run_sim(sample_size, T, arrival_rates, service_rates, eta, gamma, nash, symm_nash)

    #keep track of values for the 95% confidence interval
    emp.append(np.mean(emp_diff))

    lower_emp.append(np.percentile(emp_diff,2.5))
    upper_emp.append(np.percentile(emp_diff,97.5))

    weights.append(np.mean(weights_diff))
    
    lower_weights.append(np.percentile(weights_diff,2.5))
    upper_weights.append(np.percentile(weights_diff,97.5))


#generate figure using simulation data
sns.lineplot(x=horizons, y=emp, color='blue', label='Time average')

plt.fill_between(horizons, lower_emp, upper_emp, color='blue', alpha=0.1)

sns.lineplot(x=horizons, y=weights, color='orange', label='Strategy')

plt.fill_between(horizons, lower_weights, upper_weights, color='red', alpha=0.1)

plt.ylabel('Total variation distance from Nash equilibrium')
plt.xlabel('T')

plt.legend()
plt.show()
