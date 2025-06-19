# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# Define model parameters
class Model:
    def __init__(self):
        self.r = 0.02 # Risk-Free rate
        self.lam = 4 # Market-Price of Risk
        self.kappa = 5 # Mean-Reversion Speed
        self.theta = 0.0169 # Mean-Reversion Level
        self.beta = 0.25 # Volatility Y
        self.rho_list = [-0.4, 0, 0.4] # Correlation
        self.gam = 10 # Relative Risk-Aversion
        self.psi_list = [0.5, 1.0, 1.5] # EIS
        self.delta = 0.015 # Time-Preference Rate
        self.eps = 2 # Bequest-Motive

#Define grid parameters
class Grid:
    def __init__(self):
        self.ymin = 0.00025
        self.ymax = 0.5
        self.tmin = 0
        self.tmax = 100
        self.Ny = 100
        self.Nt = 1000000
        self.dt = (self.tmax - self.tmin) / self.Nt
        self.dy = (self.ymax - self.ymin) / self.Ny
        self.ny = np.arange(self.Ny + 1)
        self.nt = np.arange(self.Nt + 1)
        self.y = self.ymin + self.dy * self.ny

model = Model()
grid = Grid()

#Define policy function
def policy(f_ns, psi, rho):
    # Calculate numerical derivative
    f_ns_y = (np.roll(f_ns, -1) - np.roll(f_ns, 1)) / (2 * grid.dy)
    
    # Linear extrapolation at lower bound
    slope_lb = (f_ns_y[1] - f_ns_y[2]) / (grid.y[1] - grid.y[2])
    intercept_lb = f_ns_y[1] - slope_lb * grid.y[1]
    f_ns_y[0] = slope_lb*grid.y[0]+intercept_lb
    
    # Linear extrapolation at upper bound
    slope_ub = (f_ns_y[-2] - f_ns_y[-3]) / (grid.y[-2] - grid.y[-3])
    intercept_ub = f_ns_y[-2] - slope_ub * grid.y[-2]
    f_ns_y[-1] = slope_ub*grid.y[-1]+intercept_ub
    
    # Calculate optimal consumption wealth-ratio
    if psi != 1:
        theta_pref = (1-model.gam)/(1-1/psi)
        cw = (1/model.delta*f_ns**(1/theta_pref))**(-psi)
    else:
        cw = model.delta*np.ones(grid.Ny + 1)
    
    # Calculate optimal portfolio share
    pi = model.lam/model.gam+model.beta*rho/model.gam*f_ns_y/f_ns
    
    # Linear extrapolation for pi at lower bound
    slope_lb = (pi[1]-pi[2])/ (grid.y[1] - grid.y[2])
    intercept_lb= pi[1] - slope_lb* grid.y[1]
    pi[0] = slope_lb * grid.y[0] + intercept_lb
    
    # Linear extrapolation of pi at upper bound
    slope_ub = (pi[-2] - pi[-3]) / (grid.y[-2] - grid.y[-3])
    intercept_ub = pi[-2] - slope_ub * grid.y[-2]
    pi[-1] = slope_ub* grid.y[-1] + intercept_ub
    
    return cw, pi

#Define function for the coefficients of the finite difference method
def coefficients(cw, pi, rho):
    
    drift_x = model.r+model.lam*np.multiply(grid.y, pi)-0.5*model.gam*np.multiply(grid.y, pi**2)-cw
    vola = model.beta**2*grid.y/(grid.dy**2)
    corr_xy = model.beta*rho*np.multiply(grid.y, pi)/(2*grid.dy)
    drift_y = model.kappa*(model.theta-grid.y)/(2*grid.dy)
    
    coe_1 = grid.dt*((1-model.gam)*corr_xy+drift_y+0.5*vola)
    coe_2 = 1+grid.dt*((1-model.gam)*drift_x-vola)
    coe_3 = grid.dt*(-(1-model.gam)*corr_xy-drift_y+0.5*vola)
    return coe_1, coe_2, coe_3


# Define function for the aggregator value 
def aggregator(cw, f, psi):
    # Case distinction for psi
    if psi !=1:
        theta_pref = (1-model.gam)/(1-1/psi)
        value = model.delta*theta_pref*(np.multiply(cw**(1-1/psi),f**(1-1/theta_pref))-f)
    else:
        value = (1-model.gam)*model.delta*np.multiply(f, np.log(cw))
    return value


# Set up grid for saving data points
rho_max = len(model.rho_list)
psi_max = len(model.psi_list)
mult_t = 100
grid_save_data = np.arange(0, grid.Nt+1, mult_t)
Nt_save = int(grid.Nt/mult_t)
plot_t = grid.dt * grid_save_data

# Initialize value function and policies
f = np.zeros((rho_max*psi_max, grid.Ny + 1, Nt_save+1))
cw = np.zeros_like(f)
pi = np.zeros_like(f)

k=0
m=0
for rho in model.rho_list:
    print(f"Loop rho = {rho}")
    for psi in model.psi_list:
        print(f"Loop psi ={psi}")
        start_time = time.time()
        # Value function at maturity
        if psi != 1:
            theta_pref = (1-model.gam)/(1-1/psi)
            f[k, :, -1] = model.eps**((1-model.gam)/(psi-1))*model.delta**(1/theta_pref)*np.ones(grid.Ny + 1)
            f_ns_old = f[k, :, -1].copy()
        else:
            f[k, :, -1] = np.ones(grid.Ny + 1)
            f_ns_old = f[k, :, -1].copy()
            
        # Policies at maturity
        cw_ns, pi_ns = policy(f_ns_old, psi, rho)
        cw[k, :, -1] = cw_ns
        cw_ns_old = cw_ns.copy()
        pi[k, :, -1] = pi_ns
        pi_ns_old = pi_ns.copy()
        
        # Solve HJB using finite differences
        m=Nt_save-1
        for j in reversed(range(grid.Nt)):
            
            t = j * grid.dt + grid.tmin
            coe_1, coe_2, coe_3 = coefficients(cw_ns_old, psi, rho)
            
            # Compue value function a previous time step
            f_ns = coe_2 * f_ns_old + coe_1 * np.roll(f_ns_old, -1) + coe_3 * np.roll(f_ns_old, 1)+grid.dt*aggregator(cw_ns_old, f_ns_old, psi)
            
            # Linear extrapolation of f at lower bound
            slope_lb = (f_ns[1] - f_ns[2]) / (grid.y[1] - grid.y[2])
            intercept_lb = f_ns[1] - slope_lb * grid.y[1]
            f_ns[0] = slope_lb * grid.y[0] + intercept_lb
            
            # Linear extrapolation of f at upper bound
            slope_ub = (f_ns[-2] - f_ns[-3]) / (grid.y[-2] - grid.y[-3])
            intercept_ub = f_ns[-2] - slope_ub * grid.y[-2]
            f_ns[-1] = slope_ub * grid.y[-1] + intercept_ub
            
            # Save values for value function
            f_ns_old = f_ns.copy()
    
            # Save values for policy
            cw_ns, pi_ns = policy(f_ns, psi, rho)
    
            # Set values for next iteration step
            cw_ns_old = cw_ns.copy()
            pi_ns_old = pi_ns.copy()
            
            # Save values
            if j in grid_save_data:
                
                f[k, :, m] = f_ns
                cw[k, :, m] = cw_ns
                pi[k, :, m] = pi_ns
                m=m-1
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        k=k+1
print('Done Value Function Iteration')

# Plot results
colors = ['black', (0.5, 0.7, 0.2), (0.3, 0.2, 0.6)]
labels_EIS = ['EIS $= 0.5$', 'EIS $= 1.0$', 'EIS $= 1.5$']
labels_rho = [r'$\rho = -0.4$', r'$\rho= 0$', r'$\rho = 0.4$']

plt.figure()
for i in range(3):
    plt.plot(grid.y, np.log(f[i, :, 0]/f[i, 0, 0]), color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Function f(t, y) at $t=0$ for $\rho = -0.4$")
plt.xlabel("State $Y_t$")
plt.ylabel("$\ln(f(0, Y_t)/f(0, 0))$")
plt.legend()
plt.grid(False)
plt.xlim([0, 0.5])

plt.figure()
for i in range(3):
    plt.plot(grid.y, cw[i, :, 0]/cw[i, 0, 0], color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Consumption-Wealth ratio at $t=0$ for $\rho = -0.4$")
plt.xlabel(r"State $Y_t$")
plt.ylabel(r"$cw(0, Y_t)/cw(0, 0)$")
plt.legend()
plt.grid(False)
cwmax = cw[:,:, 0].max()
plt.xlim([0, 0.5])

plt.figure()
for i in range(3):
    plt.plot(grid.y, pi[i, :, 0]/pi[i, 0, 0], color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Optimal Portfolio Share at $t=0$ for $\rho = -0.4$")
plt.xlabel("State $Y_t$")
plt.ylabel(r"$\pi(0, Y_t)/\pi(0, 0)$")
plt.legend()
plt.grid(False)
plt.xlim([0, 0.5])

plt.figure()
plt.plot(grid.y, np.log(f[0, :, 0]/f[0, 0, 0]), color=colors[0], linewidth=2, label=labels_rho[0])
plt.plot(grid.y, np.log(f[3, :, 0]/f[3, 0, 0]), color=colors[1], linewidth=2, label=labels_rho[1])
plt.plot(grid.y, np.log(f[6, :, 0]/f[6, 0, 0]), color=colors[2], linewidth=2, label=labels_rho[2])
plt.title(r"Function f(t, y) at $t=0$ for EIS $=0.5$")
plt.xlabel("State $Y_t$")
plt.ylabel(r"Value function $\ln(f(0, Y_t)/f(0, 0))$")
plt.legend()
plt.grid(False)
plt.xlim([0, 0.5])

plt.figure()
plt.plot(grid.y, cw[0, :, 0]/cw[0, 0, 0], color=colors[0], linewidth=2, label=labels_rho[0])
plt.plot(grid.y, cw[3, :, 0]/cw[3, 0, 0], color=colors[1], linewidth=2, label=labels_rho[1])
plt.plot(grid.y, cw[6, :, 0]/cw[6, 0, 0], color=colors[2], linewidth=2, label=labels_rho[2])
plt.title("Consumption-Wealth ratio at $t=0$ for EIS $ = 0.5$")
plt.xlabel("State $Y_t$")
plt.ylabel("$cw(0, Y_t)/cw(0, 0)$")
plt.legend()
plt.grid(False)
plt.xlim([0, 0.5])

plt.figure()
plt.plot(grid.y, pi[0, :, 0]/pi[0, 0, 0], color=colors[0], linewidth=2, label=labels_rho[0])
plt.plot(grid.y, pi[3, :, 0]/pi[3, 0, 0], color=colors[1], linewidth=2, label=labels_rho[1])
plt.plot(grid.y, pi[6, :, 0]/pi[6, 0, 0], color=colors[2], linewidth=2, label=labels_rho[2])
plt.title(r"Optimal Portfolio Share at $t=0$ for EIS $ = 0.5$")
plt.xlabel(r"State $Y_t$")
plt.ylabel(r"$\pi(0, Y_t)/\pi(0, 0)$")
plt.ticklabel_format(useOffset=False)
plt.legend()
plt.grid(False)
plt.xlim([0, 0.5])
plt.ylim([0.9994, 1.0008])