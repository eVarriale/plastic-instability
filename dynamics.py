#!/usr/bin/env python3

import numpy as np                  # for math
import random                       # for bars problem
import pdb                          # for debugging

gain = 10.8

def sigmoidal(x, a=1., b=0.):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a * (b - x)))

def tanhyp(x, a=1., b=0.):
    return np.tanh(a * (x - b))


def simple_flow(x, φ, u, w_jk, z_jk, I_ext=0., activation=sigmoidal):
    '''
    Returns the flow without STSP
    INPUT
        x:      Membrane potential
     w_jk:      Excitatory weights
     z_jk:      Inhibitory weights

    OUTPUT
       d#:      # variation
        y:      Activity
        T:      Total input
    '''

    y = activation(x, gain, 0.)
    rec_input = np.dot(w_jk+z_jk, y)
    T = rec_input + I_ext

    dx = (T - x) / T_x

    du = 0 # Needed to have the same output as STSP flow
    dφ = 0 # Same

    return dx, du, dφ, y, T


# Tsodyks-Markram parameters
U_m = 18
# 4 from Bulcsù's notes, ER network
# 18 to get u \in [1, 4]
T_u = 21.
# 21 ms from Bulcsù's notes, (Gupta et al. 2000 F2 GABA)
# 200 ms from Mongillo '08 Supplementary
T_φ = 706.
# 706 ms from Bulcsù's notes, (Gupta et al. 2000 F2 GABA)
α = 0.01
# 10 s^-1 = 0.01 ms^-1
T_x = 20.

# Asymptotic values for y = y_inf
y_inf = 1
u_inf = (1 + α * U_m * T_u * y_inf)/(1 + α * T_u * y_inf)
φ_inf = 1 / (1 + α * u_inf * T_φ * y_inf)
uφ_inf = u_inf * φ_inf

def tsodyks_markram(x, φ, u, w_jk, z_jk, I_ext=0., activation=sigmoidal):
    '''
    Returns the flow of the Tsodyks-Markram model
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

    du = (1. - u) / T_u + α * (U_m - u) * y
    #    relax to 1      driven to U_max by firing
    dφ = (1. - φ) / T_φ - α * φ * u * y
    #  replenish vesicles  empty when firing
    return dx, du, dφ, y, T


U_max_inh = 4.
T_u_inh = 30. # 10.
T_φ_inh = 60. # 20.

def full_depletion(x, φ, u, w_jk, z_jk, I_ext=0., activation=sigmoidal):
    '''
    Returns the flow of the full depletion model
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
    y = activation(x, gain, 0)
    eff_y = y * φ * u
    # This enforces Tsodyks-Markram rule I_j = sum_k z_jk phi_k u_k y_k
    exc_input = np.dot(w_jk, y)
    inh_input = np.dot(z_jk, eff_y)
    tot_input = inh_input + exc_input + I_ext

    dx = (tot_input - x) / T_x

    U_y = 1 + (U_max_inh - 1) * y
    d_u = (U_y - u) / T_u_inh

    ϕ_u = 1 - u * y / U_max_inh
    d_φ = (ϕ_u - φ) / T_φ_inh

    return dx, d_u, d_φ, y, tot_input

def int_arctanh(x):
    ''' Returns the int_0^x arctanh '''
    return 0.5 * np.log(1 - x**2) + x * np.arctanh(x)

def cont_hopfield_energy(weights, state, ext_input):
    E = -0.5 * state @ weights @ state - ext_input @ state + int_arctanh(state).sum()
    return E
