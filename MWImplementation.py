import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

class Player:
    def __init__(self, name, actionSet, learningRate):
        self.name = name
        self.learningRate = learningRate
        self.weights = np.array([1/(len(actionSet)) for i in range(len(actionSet))])
        self.actions = actionSet
    def normalizeWeights(self):
        self.weights = self.weights / sum(self.weights)

    def getAction(self):
        self.normalizeWeights()
        return np.random.choice(self.actions, p=self.weights)

    def updateWeights(self, costVector):
        updates = np.array([1 for i in range(len(costVector))])-self.learningRate*costVector
        self.weights = self.weights * updates
        self.normalizeWeights()

class Adversary:
    def __init__(self, name):
        self.name = name

    def getCosts(self, probabilities):
        randomness = np.random.rand(len(probabilities))
        return np.array([1 for i in range(len(probabilities))])-2*probabilities + randomness #most cost for most probable choice, semi-smart adversary

X = []
Y = []

possible_time_spans = [500*i for i in range(1, 41)]
for time_span in possible_time_spans:
    T = time_span
    n = 10
    learning_rate = math.sqrt(math.log(n) / T)
    actions = range(n)
    player = Player('Player1', actions, learning_rate)
    adversary = Adversary('Adversary1')
    costs = np.zeros((T, n))
    totalCost = 0
    for i in range(T):
        player.getAction()
        costs[i] = adversary.getCosts(player.weights)
        totalCost += sum(costs[i] * player.weights)
        player.updateWeights(costs[i])

    transposed_costs = np.transpose(costs)
    fixed_costs = [sum(transposed_costs[i]) for i in range(n)]
    best_fixed = min(fixed_costs)

    regret = (totalCost - best_fixed) / T
    #print("Time span T: "+str(T)+". Regret: "+str(regret))
    X.append(time_span)
    Y.append(regret)

X = np.array(X)
Y = np.array(Y)
plt.scatter(X, Y)
plt.title("MW Algorithm: Regret versus Time Span")
plt.xlabel("Time Span")
plt.ylabel("Regret")
plt.show()


X_prime = 1/np.sqrt(X)
plt.scatter(X_prime, Y)
plt.title("Regret vs. 1/sqrt(T)")
plt.xlabel("1/sqrt(T)")
plt.ylabel("Regret")
plt.show()