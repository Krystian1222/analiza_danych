# Variable selection using information gain.

import pandas as pd
import numpy as np
from scipy.sparse import random
from scipy import stats

# Uploading a file zoo.csv
data = pd.read_csv('zoo.csv')

# 1 - Write a function that, for a given column of discrete data 'x', returns: the unique values of 'x_i', their estimated probabilities 'p_i' or frequencies 'n_i'.
def freq(x, prob=True):
    x = np.array(x)
    p_i = 0
    x_i, n_i = np.unique(x, return_counts=True)
    if prob == True:
        p_i = n_i / len(x)
        return x_i, p_i
    return x_i, n_i

example_dict = dict()
example_dict['numbers'] = [1, 2, 3, 2, 1, 2, 4, 6]
print('Task 1. Function validation.')
z = freq(example_dict['numbers'], True)
print(z)

# 2 - Write a function that for given data columns 'x' and 'y' returns: unique attribute values 'xi', 'yi' and total frequency or count distribution 'ni' (depending on parameter 'prob').
def freq2(x, y, prob = True):
    if prob == False:
        xi, ni_x = np.unique(x, return_counts = True)
        yi, ni_y = np.unique(y, return_counts = True)
        return xi, yi, ni_x, ni_y
    else:
        xi = np.unique(x).tolist()
        yi = np.unique(y).tolist()
        pi = np.zeros((len(xi), len(yi)))
        for i in range(0, len(x)):
            u = xi.index(x[i])
            v = yi.index(y[i])
            pi[u, v] = pi[u, v] + 1
        return xi, yi, pi/len(x)

# Function to calculate mutual information.
def freq2_1(frame, k1, k2, prob = True):
    if isinstance(frame, pd.DataFrame) == False:
        return None
    else:
        if k1 in frame and k2 in frame and k1 != k2:
            matrix = frame.groupby([k1, k2])
            matrix = matrix.size()
            matrix = matrix.unstack(fill_value=0)
            if prob == True:
                sum = matrix.sum().sum()
                matrix = matrix / sum
            return matrix

print('\nTask 2. Function validation.')
x = freq2(data["feathers"], data["domestic"], prob=False)
print(x)

# 3 - Write functions that calculate entropy and mutual information gain.
# entropy
def entropy(x):
    x = x[x > 0]
    return -np.sum(x*np.log2(x))

# information gain
def infogain(x, y):
    if y != 'animal':
        p = np.array(freq2_1(x, 'animal', y, prob=True))
        H_X = entropy(np.sum(p, axis=1))
        H_Y = entropy(np.sum(p, axis=0))
        H_XY = entropy(p)
        return H_X + H_Y - H_XY

print('\nTask 3.')
print('Entropy:')
for i in data:
    a, b = freq(data[i])
    print(entropy(b))

print('Information gain:')
for i in data.keys():
    print(infogain(data, i))

# Auxiliary functions calculating entropy (H) and mutual information (I)
H = lambda p: -np.sum(p[p > 0] * np.log2(p[p > 0]))
I = lambda p: H(np.sum(p, axis=0)) + H(np.sum(p, axis=1)) - H(p)

# 4 - Load test data and perform attribute selection/ranking using the information gain criterion.
print('\nTask 4.')
df = pd.DataFrame(columns = ['Columns', 'Mutual information'])
y = 0
for i in data:
    for o in data:
        q, w, b = freq2(data[i], data[o])
        df.loc[y] = [[str(i), str(o)], [I(b)]]
        y += 1
df = df.sort_values(by='Mutual information')
print(df)

# 5 - Check that the functions 'freq', 'freq2' work for sparse attributes.
# Class to generate the random_state parameter.
class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs
S = random(15, 15, density=0.1, random_state=rs, data_rvs=rvs)
S = S.A
print("\nTask 5.")
print("Print of sparse matrix:")
print(S)
example_dict["matrix"] = S[:, 1]
print("Operation of the freq function for sparse attributes - second column send:")
z = freq(example_dict["matrix"], prob=False)
print(z)
print("Operation of the freq2 function for sparse attributes - second and fifth column send:")
x = freq2(S[:, 1], S[:, 4], prob=False)
print(x)
