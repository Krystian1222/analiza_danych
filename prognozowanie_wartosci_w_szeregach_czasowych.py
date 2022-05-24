# Prognozowanie wartości w szeregach czasowych na przykładzie danych dotyczących epidemii COVID-19

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import brenth

def diff_fun(fun, h = 1e-7):
    return lambda x: ((fun(x + h) - fun(x - h)) / 2 / h) - 1

# 9 - Zdefiniowanie funkcji 'gompertz'.
def gompertz(t, N0, b, c):
    return N0 * np.exp((-1) * b * np.exp((-1) * c * t))

# 1 - Wczytanie danych dotyczących rozwoju epidemii COVID-19 (odpowiednio przypadki potwierdzone, śmiertelne, wyleczone) do trzech tabel.
dane_1 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
dane_2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
dane_3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
dc = pd.read_csv(dane_1)
dd = pd.read_csv(dane_2)
dr = pd.read_csv(dane_3)

# 2 - Dla każdej z tabel ustawienie indeksu na czterech początkowych zmiennych.
dc = dc.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dd = dd.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dr = dr.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])

# 3 - Zwalcowanie każdej z tabel za pomocą metody stack.
dc = dc.stack()
dd = dd.stack()
dr = dr.stack()

# 4 - Złączenie uzyskanych trzech serii danych w jedną tabelę.
df = pd.DataFrame({'c': dc, 'd': dd, 'r': dr})

# 5 - Usunięcie indeksu za pomocą funkcji reset_index()
df = df.reset_index()

# 6 - Dodanie nowej kolumny o nazwie 'czas' zawierającej poprawnie sparsowaną datę.
level_4 = pd.to_datetime(df['level_4'])
df = df.assign(data = level_4)
df['czas'] = pd.to_datetime(df['level_4'])

# 7 - Utworzenie nowej kolumny 't', która będzie przechowywała czas w dniach od ustalonej daty.
df['t'] = ((df['czas'] - pd.Timestamp('2020/03/01')) / np.timedelta64(1, 'D')).astype(int)
df = df.assign(diff = ((level_4 - pd.to_datetime("2020/03/01"))/ np.timedelta64(1, 'D')).astype(int))

# 8 - Wybranie danych dotyczących Polski.
pol = df[df['Country/Region'] == 'Poland']
polska = pol.assign(cdiff = pol['c'].diff())

d = pol['c'].diff()
y = np.asarray(pol['c'])
x = np.asarray(pol['t'])
plt.xlabel('Czas [dni]')
plt.ylabel('Liczba zachorowań')
plt.title('Zadanie 8. Skumulowana liczba przypadków na dany dzień.')
plt.plot(x, y, 'o')
plt.savefig("Zadanie_8_liczba_przypadkow_dziennie.png")
plt.figure()
plt.xlabel('Czas [dni]')
plt.ylabel('Przyrost zachorowań')
plt.title('Zadanie 8. Przyrosty dzienne.')
plt.plot(x, d, 'o')
plt.savefig("Zadanie_8_liczba_przyrostow_dziennych.png")

# 10 - Dopasowanie do danych krzywej Gompertza za pomocą funkcji curve_fit
x = pol['diff'].to_numpy()
y = pol['c'].to_numpy()
f = np.vectorize(gompertz)
t = np.linspace(-40, 120, 161)
tp = np.linspace(-40, 400, 200)

popt, pcov = curve_fit(f, x, y, p0 = [1., 1., 1.])
ym = f(t, *popt)
ymp = f(tp, *popt)
plt.figure()
plt.xlabel('Czas [dni]')
plt.ylabel('Liczba zachorowań')
plt.title('Zadanie 10. Skumulowana liczba zachorowań.')
plt.scatter(pol['diff'], pol['c'])
plt.plot(t, ym, 'g-')
plt.savefig("Zadanie_10_skumulowana_liczba_zachorowan.png")
plt.figure()
plt.xlabel('Czas [dni]')
plt.ylabel('Liczba zachorowań na dzień')
plt.title('Zadanie 10. Pochodna z dopasowanej krzywej Gompertza.')
plt.scatter(pol['diff'], polska['cdiff'])
plt.plot(t, diff_fun(lambda x: f(x, *popt))(t), 'r-')
plt.savefig("Zadanie_10_pochodna_z_gompertza.png")

# 11 - Odczytanie prognozowaną łączną liczbę zachorowań na COVID-19 w Polsce. Określenie, kiedy model przewiduje szczyt zachorowań, kiedy będzie prognozowany koniec epidemii oraz kiedy będzie połowa epidemii.
root = brenth(diff_fun(lambda x: f(x, *popt)), 0, 500)
print("Dzień końca epidemii: {}.".format(m.ceil(root)))

max_l_zach = m.ceil(f(m.ceil(root), *popt))
print('Maksymalna liczba zachorowań: {}.'.format(max_l_zach))

zach_50 = 0.5 * max_l_zach
polowa_epidemii = m.ceil(np.interp(zach_50, ymp, tp))
print('Połowa epidemii: {} dzień.'.format(polowa_epidemii))

zach_99 = 0.99 * max_l_zach
prawie_koniec = m.ceil(np.interp(zach_99, ymp, tp))
print('99% epidemii: {} dzień.'.format(int(prawie_koniec)))
plt.show()
