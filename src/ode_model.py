from dataclasses import dataclass
import numpy as np
#import matplotlib.pyplot as plt

@dataclass
class ODEParams:
    alpha: float = 1.0   # prey birth rate
    beta: float = 0.1    # predation - how often predators encounter prey
    delta: float = 0.075 # efficiency - how well predators convert prey to new predators
    gamma: float = 1.5   # predator death
    dt: float = 0.01 # time step (increment)
    t_end: float = 50.0 # end time of simulation

# Lotka-Volterra model simulation
def lotka_volterra(x0=40., y0=9., params: ODEParams = ODEParams()):
    # x: prey, y: pred
    n_steps = int(params.t_end / params.dt)
    X = np.zeros(n_steps+1); Y = np.zeros(n_steps+1); T = np.zeros(n_steps+1)
    X[0] = x0; Y[0] = y0
    for i in range(n_steps):
        x, y = X[i], Y[i]
        dx = params.alpha*x - params.beta*x*y
        dy = params.delta*x*y - params.gamma*y
        X[i+1] = max(0.0, x + params.dt*dx)
        Y[i+1] = max(0.0, y + params.dt*dy)
        T[i+1] = T[i] + params.dt
    return T, X, Y

def statistics(X, Y):
    max_prey = np.max(X)
    min_prey = np.min(X)
    max_pred = np.max(Y)
    min_pred = np.min(Y)
    return max_prey, min_prey, max_pred, min_pred
