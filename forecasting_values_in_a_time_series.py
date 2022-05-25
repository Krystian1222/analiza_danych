# Forecasting values in time series using COVID-19 epidemic data as an example.

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import brenth

def diff_fun(fun, h = 1e-7):
    return lambda x: ((fun(x + h) - fun(x - h)) / 2 / h) - 1

# 9 - definition of 'gompertz' function.
def gompertz(t, N0, b, c):
    return N0 * np.exp((-1) * b * np.exp((-1) * c * t))

# 1 - Load data on the evolution of the COVID-19 epidemic (confirmed, fatal, cured cases, respectively) into three tables.
data_1 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data_2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
data_3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
dc = pd.read_csv(data_1)
dd = pd.read_csv(data_2)
dr = pd.read_csv(data_3)

# 2 - For each table, setting the index on the four initial variables.
dc = dc.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dd = dd.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])
dr = dr.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'])

# 3 - Collate each table using the 'stack()' method.
dc = dc.stack()
dd = dd.stack()
dr = dr.stack()

# 4 - Combine the resulting three data series into a single table.
df = pd.DataFrame({'c': dc, 'd': dd, 'r': dr})

# 5 - To delete an index using the reset_index() function.
df = df.reset_index()

# 6 - Adding a new column called 'czas' containing a correctly sparsified date.
level_4 = pd.to_datetime(df['level_4'])
df = df.assign(data=level_4)
df['czas'] = pd.to_datetime(df['level_4'])

# 7 - Creation of a new column 't' which will store the time in days since the set date.
df['t'] = ((df['czas'] - pd.Timestamp('2020/03/01')) / np.timedelta64(1, 'D')).astype(int)
df = df.assign(diff=((level_4 - pd.to_datetime("2020/03/01"))/ np.timedelta64(1, 'D')).astype(int))

# 8 - Selecting data on Poland.
pol = df[df['Country/Region'] == 'Poland']
poland = pol.assign(cdiff=pol['c'].diff())

d = pol['c'].diff()
y = np.asarray(pol['c'])
x = np.asarray(pol['t'])
plt.xlabel('Time [days]')
plt.ylabel('Number of cases')
plt.title('Task 8: Cumulative number of cases per day.')
plt.plot(x, y, 'o')
plt.savefig("Task_8_number_of_cases_daily.png")
plt.figure()
plt.xlabel('Time [days]')
plt.ylabel('Increase in incidence')
plt.title('Task 8. Daily increments.')
plt.plot(x, d, 'o')
plt.savefig("Task_8_number_of_daily_increments.png")

# 10 - Fitting a Gompertz curve to data using the curve_fit() function.
x = pol['diff'].to_numpy()
y = pol['c'].to_numpy()
f = np.vectorize(gompertz)
t = np.linspace(-40, 120, 161)
tp = np.linspace(-40, 400, 200)

popt, pcov = curve_fit(f, x, y, p0=[1., 1., 1.])
ym = f(t, *popt)
ymp = f(tp, *popt)
plt.figure()
plt.xlabel('Time [days]')
plt.ylabel('Number of cases')
plt.title('Task 10. Cumulative incidence.')
plt.scatter(pol['diff'], pol['c'])
plt.plot(t, ym, 'g-')
plt.savefig("Task_10_cumulative_incidence.png")
plt.figure()
plt.xlabel('Time [days]')
plt.ylabel('Number of cases per day')
plt.title('Task 10. The derivative of the fitted Gompertz curve.')
plt.scatter(pol['diff'], poland['cdiff'])
plt.plot(t, diff_fun(lambda x: f(x, *popt))(t), 'r-')
plt.savefig("Task_10_derivative_of_the_Gompertz.png")

# 11 - Reading the projected total number of COVID-19 cases in Poland. Determine when the model predicts the peak incidence, when the end of the epidemic will be predicted and when the halfway point of the epidemic will be.
root = brenth(diff_fun(lambda x: f(x, *popt)), 0, 500)
print("The day the epidemic ends: {}.".format(m.ceil(root)))

max_number_of_cases = m.ceil(f(m.ceil(root), *popt))
print('Maximum number of cases: {}.'.format(max_number_of_cases))

noc_50 = 0.5 * max_number_of_cases
half_of_the_epidemic = m.ceil(np.interp(noc_50, ymp, tp))
print('Half of the epidemic: {} day.'.format(half_of_the_epidemic))

noc_99 = 0.99 * max_number_of_cases
prawie_koniec = m.ceil(np.interp(noc_99, ymp, tp))
print('99% of the epidemic: {} day.'.format(int(prawie_koniec)))
plt.show()
