import numpy as np
import math

def elf(m, n, theta, epsilon):
    y = theta + math.sqrt(epsilon)
    numX = np.random.binomial(m, theta) # number of 1s

    scoregap0, scoregap1 = y**2 - theta**2, (1-y)**2 - (1-theta)**2 # R(theta, x) - R(y, x)
    f0 = np.array([(1 + scoregap0)] + [(1 - scoregap0/(n-1))]*(n-1))/n # probs for x=0
    f1 = np.array([(1 + scoregap1)] + [(1 - scoregap1/(n-1))]*(n-1))/n # probs for x=1

    winners = np.append(np.random.choice(n, size=m-numX, p=f0), np.random.choice(n, size=numX, p=f1))
    return np.argmax(np.bincount(winners))

def many_elf(m, n, theta, epsilon, reps):
    return 1 - np.count_nonzero([elf(m, n, theta, epsilon) for i in range(reps)])/float(reps)

def elf_bound(m, n, epsilon):
    return 1 - 4*(n-1)*math.exp((0-m*epsilon**2)/(2*(n-1)**2))

def shelf(m, n, theta, epsilon):
    y = theta + math.sqrt(epsilon)
    numX = np.random.binomial(m, theta) # number of 1s
    truthfulScore = np.random.binomial(m-numX,1-theta**2) + np.random.binomial(m-numX, 1-(1-theta)**2)
    otherScores = [np.random.binomial(m-numX,1-y**2) + np.random.binomial(m-numX, 1-(1-y)**2) for i in range(n-1)]
    return np.argmax([truthfulScore] + otherScores)

def many_shelf(m, n, theta, epsilon, reps):
    return 1 - np.count_nonzero([shelf(m, n, theta, epsilon) for i in range(reps)])/float(reps)

def shelf_bound(m, n, epsilon):
    return 1 - 4*(n-1)*math.exp(0-(m*epsilon**2)/2)

