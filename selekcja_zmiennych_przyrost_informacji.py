# Selekcja zmiennych za pomocą przyrostu informacji.

import pandas as pd
import numpy as np
from scipy.sparse import random
from scipy import stats

# Wczytanie pliku zoo.csv
dane = pd.read_csv('zoo.csv')

# 1 - Napisanie funkcji, która dla zadanej kolumny danych 'x' dyskretnych zwróci: unikalne wartości 'x_i', ich estymowane prawdopodobieństwa 'p_i' lub częstości 'n_i'.
# x_i - unikalne wartości
# p_i - estymowane prawdopodobieństwa
# n_i - częstości
def freq(x, prob = True):
    x = np.array(x)
    p_i = 0
    x_i, n_i = np.unique(x, return_counts = True)
    if prob == True:
        p_i = n_i / len(x)
        return x_i, p_i
    return x_i, n_i

slownik = {}
slownik["liczby"] = [1, 2, 3, 2, 1, 2, 4, 6]
print("Zadanie 1. Sprawdzenie poprawności funkcji:")
z = freq(slownik["liczby"], True)
print(z)

# 2 - Napisanie funkcji, która dla zadanych kolumn danych 'x' i 'y' zwróci: unikalne wartości atrybutów 'xi', 'yi' oraz łączny rozkład częstości lub liczności 'ni' (w zależności od parametru 'prob').
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

# funkcja umożliwiająca obliczenie informacji wzajemnej
def freq2_1(ramka, k1, k2, prob = True):
    if isinstance(ramka, pd.DataFrame) == False:
        return None
    else:
        if k1 in ramka and k2 in ramka and k1 != k2:
            macierz = ramka.groupby([k1, k2])
            macierz = macierz.size()
            macierz = macierz.unstack(fill_value = 0)
            if prob == True:
                suma = macierz.sum().sum()
                macierz = macierz / suma
            return macierz

print("\nZadanie 2. Sprawdzenie poprawności funkcji:")
x = freq2(dane["feathers"], dane["domestic"], prob = False)
print(x)

# 3 - Napisanie funkcji, które obliczą entropię oraz przyrost informacji wzajemnej.
# entropia
def entropy(x):
    x = x[x > 0]
    return -np.sum(x*np.log2(x))

# informacja wzajemna
def infogain(x, y):
    if y != 'animal':
        p = np.array(freq2_1(x, 'animal', y, prob = True))
        H_X = entropy(np.sum(p, axis = 1))
        H_Y = entropy(np.sum(p, axis = 0))
        H_XY = entropy(p)
        return H_X + H_Y - H_XY

print("\nZadanie 3.")
print("Entropia:")
for i in dane:
    a, b = freq(dane[i])
    print(entropy(b))

print("Informacja wzajemna:")
for i in dane.keys():
    print(infogain(dane, i))

# funkcje pomocnicze obliczające entropię (H) oraz informację wzajemną (I)
H = lambda p: -np.sum(p[p>0]*np.log2(p[p>0]))
I = lambda p:H(np.sum(p,axis=0))+H(np.sum(p,axis=1)) - H(p)

# 4 - Wczytanie danych testowych oraz dokonanie selekcji/stopniowania atrybutów z wykorzystaniem kryterium przyrostu informacji.
print("\nZadanie 4.")
df = pd.DataFrame(columns = ['Kolumny', 'Informacja wzajemna'])
y = 0
for i in dane:
    for o in dane:
        q,w,b = freq2(dane[i], dane[o])
        df.loc[y] = [[str(i), str(o)], [I(b)]]
        y += 1
df = df.sort_values(by = 'Informacja wzajemna')
print(df)

# 5 - Sprawdzenie, czy funkcje 'freq', 'freq2' działają dla atrybutów rzadkich.
# klasa do generowania parametru random_state
class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.poisson(25, loc = 10).rvs
S = random(15, 15, density = 0.1, random_state = rs, data_rvs = rvs)
S = S.A
print("\nZadanie 5.")
print("Wypis macierzy rzadkiej:")
print(S)
slownik["macierz"] = S[:, 1]
print("Działanie funkcji freq dla atrybutów rzadkich - przesłanie drugiej kolumny:")
z = freq(slownik["macierz"], prob = False)
print(z)
print("Działanie funkcji freq2 dla atrybutów rzadkich - przesłanie drugiej i piątej kolumny:")
x = freq2(S[:, 1], S[:, 4], prob = False)
print(x)