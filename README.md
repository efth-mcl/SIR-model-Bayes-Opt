# SIR-model-Bayes-Opt
SIR model for COVID-19 using Bayesian Optimization

```python
!pip install bayesian-optimization
!git clone https://github.com/CSSEGISandData/COVID-19.git
```


```python
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import pandas as pd
```


```python
df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
rec = df[df['Country/Region']=="Greece"].values.T[4:].reshape(-1)
```


```python
df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
conf = df[df['Country/Region']=="Greece"].values.T[4:].reshape(-1)
```


```python
wg0 = np.where(conf>0)
True_rec = rec[wg0]
plt.figure(figsize=(8,4))
plt.plot(True_rec)
plt.xlabel('Days')
plt.ylabel('Data Recovered')
```




    Text(0, 0.5, 'Data Recovered')




![png](output_4_1.png)



```python
def SIR(t, sir, N, b, g):
    S, I, R = sir
    return [-b*S*I/N, b*S*I/N-g*I, g*I ]
```


```python
%matplotlib inline
from ipywidgets import interactive
Pop = 3.756*10**6*0.001
Pop = 4061
D_T = True_rec.shape[0]
T = D_T+20
t = np.linspace(0, T, T)
d_t = np.linspace(0,D_T,D_T)
def f(b, g,Pop):
    plt.figure(figsize=(16,8))
    sol_ep = solve_ivp(SIR, [0, T], [0.999*Pop, 0.001*Pop, 0], args=(Pop, b, g), dense_output=True)
    ep = sol_ep.sol(t)

    plt.plot(t, ep.T[:,0], label = 'Susceptible')
    plt.plot(t, ep.T[:,1],label='Infected')
    plt.plot(t, ep.T[:,2], label='Recovered')
    plt.plot(True_rec, label='True Data')
    plt.xlabel('t')
    plt.legend(shadow=True)
    loss = (Pop*ep.T[:D_T,2]-True_rec)*np.linspace(0.001,1,D_T)
    loss = (loss-np.mean(loss))/np.std(loss)
    loss = np.mean(np.log(np.cosh(list(loss))))
    plt.title('SIR Greece, {}'.format(loss))
    plt.ylabel('Population')
    plt.show()

interactive_plot = interactive(f, b=(0.1,0.5,0.001), g=(0.01,0.05,0.001),Pop=(1*10**3,5*10**3))
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot
```


    interactive(children=(FloatSlider(value=0.30000000000000004, description='b', max=0.5, min=0.1, step=0.001), F…



```python
D_T = True_rec.shape[0]
d_t = np.linspace(0,D_T,D_T)
std = np.std(True_rec)
mn = np.mean(True_rec)
n_data = (True_rec-mn)/std
F = 1
def black_box_function(Pop, b, g):
    sol_ep = solve_ivp(SIR, [0, D_T], [0.999*Pop, 0.001*Pop, 0], args=(Pop, b, g), dense_output=True)
    ep = sol_ep.sol(d_t)
    d = (ep.T[:,2]-mn)/std
    d = (d-n_data)*np.linspace(0.001,F,D_T)
    loss = np.mean(np.log(np.cosh(list(d))))
    return -loss
```


```python
pbounds = {'Pop': (conf[-1]*1.2, 10*10**3),'b':(0.1,0.5), 'g':(0.01,0.05)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)


```


```python
optimizer.maximize(init_points=50, n_iter=50)
```


```python
optimizer.max
```




    {'params': {'Pop': 4207.066585791539,
      'b': 0.15950142917185214,
      'g': 0.026386258654299888},
     'target': -0.01625127762692438}




```python
%matplotlib inline
from ipywidgets import interactive
Pop = optimizer.max['params']['Pop']
D_T = True_rec.shape[0]
T = D_T+150
t = np.linspace(0, T, T)
d_t = np.linspace(0,D_T,D_T)
b = optimizer.max['params']['b']
g = optimizer.max['params']['g']
plt.figure(figsize=(16,8))
sol_ep = solve_ivp(SIR, [0, T], [0.999*Pop, 0.001*Pop, 0], args=(Pop, b, g), dense_output=True)
ep = sol_ep.sol(t)

plt.plot(t, ep.T[:,0], label = 'Susceptible')
plt.plot(t, ep.T[:,1],label='Infected')
plt.plot(t, ep.T[:,2], label='Recovered')
plt.plot(True_rec, label='True Recoverd')
plt.xlabel('t')
plt.legend(shadow=True)


d = (ep.T[:D_T,2]-mn)/std
d = (d-n_data)*np.linspace(0.001,F,D_T)
loss = np.mean(np.log(np.cosh(list(d))))

plt.title('SIR Loss, {}'.format(loss))
plt.ylabel('Population')
plt.show()
```


![png](output_11_0.png)



```python
print('b', b)
print('g', g)
print('R0', b/g)
print('Pop', Pop)
```

    b 0.15950142917185214
    g 0.026386258654299888
    R0 6.044867188697094
    Pop 4207.066585791539


# References
* COVID-19 Data: https://github.com/CSSEGISandData/COVID-19
* SIR model: https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
* Bayesian Optimization: https://github.com/fmfn/BayesianOptimization
