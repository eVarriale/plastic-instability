#!/usr/bin/env python3

import numpy as np                  # for math
import random                       # for bars problem
import pdb                          # for debugging


gain = 0.8

def activation(x, a=1., b=0.):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a*(b - x)))


# Tsodyks-Markram parameters
U_max = 4.
T_u = 21. #10.
T_φ = 706. #20.
α = 10./1000. # 10 s^-1
T_x = 20.

# Asymptotic values for y = y_inf
y_inf = 1
u_inf = (1 + α * U_max * T_u * y_inf)/(1 + α * T_u * y_inf)
φ_inf = 1 / (1 + α * u_inf * T_φ * y_inf)
uφ_inf = u_inf * φ_inf

def tsodyks_markram(x, φ, u, w_jk, z_jk, I_ext=0.):

    ''' 
    Next step with Euler integration 
    INPUT
        x:      Membrane potential
        φ:      Full vesicles
        u:      Vesicle release likelihood
     w_jk:      Excitatory weights
     z_jk:      Inhibitory weights

    OUTPUT
       d#:      # variation
        y:      Activity
        T:      Total input
    '''
    y = activation(x, gain, 0.)
    y_eff = y * φ * u
    # This enforces Tsodyks-Markram rule I_j = sum_k z_jk phi_k u_k y_k
    excit_input = np.dot(w_jk, y)
    inhib_input = np.dot(z_jk, y_eff)
    T = excit_input + inhib_input + I_ext

    dx = (T - x) / T_x

    du = (1. - u) / T_u + α * (U_max - u) * y
    #    relax to 1      driven to U_max by firing
    dφ = (1. - φ) / T_φ - α * φ * u * y
    #  replenish vesicles  empty when firing   
    return dx, du, dφ, y, T