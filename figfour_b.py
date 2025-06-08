import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

#implements the simulation for figures 4a,b using the EXP3 algorithm
def run_sim(sample_size, T, arrival_rates, service_rates, eta, gamma, x_len):

    n = len(arrival_rates)
    m = len(service_rates)

    emp_tracker = [[[] for j in range(m)] for i in range(n)]
    weight_tracker = [[[] for j in range(m)] for i in range(n)]


    for sample in range(sample_size):

        emp = np.zeros((n,m))
        sent = np.zeros(n)

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

            #packets-sending logic
            for j in range(n):
                if arrived[j] > 0: queue_packets[j] += 1
                if queue_packets[j]: 
                    sent[j] += 1 
                    deviate = np.random.binomial(size=1,n=1,p=gamma)[0]
                    choice = rng.choice(m) if deviate else rng.choice(m,p=weights[j])
                    choices[j] = choice
                    emp[j][choice] += 1
                    server_packets[choice].append(j)

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

            #keep track of queues' strategies
            if t % x_len == 0:
                for i in range(n):
                    for j in range(m):
                        emp_tracker[i][j].append(emp[i][j]/sent[i])
                        weight_tracker[i][j].append(weights[i][j])
    return emp_tracker, weight_tracker

#hardcoded values for the setup in the paper

plt.rcParams.update({'font.size': 14})

sample_size = 1

T = 1_000_000

eta = 1/(np.sqrt(T))

gamma = eta

x_len = 100

points = T//x_len

arrival_rates = [1/4,1/4]

service_rates = [2/3,2/3]

m = len(service_rates)

nash = [[[0]*points,[1]*points],[[1]*points,[0]*points]]
symm_nash = [[[1]*points,[0]*points],[[0]*points,[1]*points]]
mixed_nash = [[[1/2]*points,[1/2]*points],[[1/2]*points,[1/2]*points]]

weights = []

lower_weights = []

upper_weights = []

emp = []

lower_emp = []

upper_emp = []

#run simulation
emp, weights = run_sim(sample_size, T, arrival_rates, service_rates, eta, gamma, x_len)

#generate figure using simulation data
x_axis = [x_len*i for i in range(points)]
linestyles = ["-", "--", ":"]
colors = ["blue", "orange", "red", "brown"]

for i in range(len(arrival_rates)):
    plt.plot(x_axis, nash[i][0], linestyle=linestyles[2], color="gray")
    plt.plot(x_axis, symm_nash[i][0], linestyle=linestyles[2], color="gray")
    plt.plot(x_axis, mixed_nash[i][0], linestyle=linestyles[2], color="gray")
    plt.plot(x_axis, emp[i][0], linestyle=linestyles[1], color=colors[i])
    plt.plot(x_axis, (1-gamma)*np.array(weights[i][0]) + gamma/m, linestyle=linestyles[0], color=colors[i])


plt.ylabel('Probability of sending a packet')
plt.xlabel('Time')
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Q1 to S1'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Q2 to S1'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Nash Eq')
]

plt.legend(handles=legend_elements,  fontsize=12, loc="best")

plt.show()
