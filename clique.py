#!/usr/bin/env python3

import importlib                    # for reimporting libraries
import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from tqdm import tqdm               # for progress bar
import os                           # for saving stuff
import shutil                       # for copying source code
#import pdb                          # for debugging
#pdb.set_trace()

#plt.style.use('seaborn-poster')
#plt.style.use('seaborn-talk')
plt.style.use('seaborn-ticks')
import networks as net
import dynamics
import plotting as myplot

plt.ion()

def reload_mod():
    importlib.reload(net)
    importlib.reload(dynamics)
    importlib.reload(myplot)

# Global variables

sim_steps = 1 * 10 * 1000 # integration steps
delta_t = 1 # integration time step in milliseconds
trials_num = 1
w_mean = 1

n_clique = 4
clique_size = 6 # >2 !, otherwise cliques don't make sense
n_links = 1

#seed = 7 
#np.random.seed(seed)

# Initialize stuff


#weights, graph, G_exc = ErdosRenyi_network()
#w_jk, z_jk, graph = net.single_neuron()
w_jk, z_jk, graph = net.scaling_network(n_clique)
final_times = np.minimum(sim_steps, 10000)
time_plot = np.arange(sim_steps-final_times, sim_steps)/int(1000/delta_t)
# time in seconds


how_many_winning_cliques = []
final_states = []
which_clique_won = []

for trial in tqdm(range(trials_num)):

    #w_jk, z_jk, graph = net.scaling_network(n_clique, n_links=n_links, w_scale=1)
    weights = w_jk + z_jk
    neurons = graph.number_of_nodes()
    neurons_over_time = (neurons, sim_steps)
    neurons_plot = neurons

    x = np.random.uniform(-1, 1, neurons)
    #x = np.zeros(neurons)
    #x[graph.clique_list[0]] = np.random.normal(0.6, 0.1, graph.s_c)
    #x[graph.clique_list[1]] = np.random.normal(1, 0.1, graph.s_c)
    #first_clique = np.ones(clique_size)
    #x[graph.clique_list[0]] = 10.

    activity_record = np.zeros(neurons_over_time)
    input_record = np.zeros(neurons_over_time)
    membrane_pot_record = np.zeros(neurons_over_time)


    φ = 1 * np.random.normal(1, 0.1, neurons)
    u = 1 * np.random.normal(1, 0.1, neurons)
    full_vesicles_record = np.zeros(neurons_over_time)
    vesic_release_record = np.zeros(neurons_over_time)

    # Main simulation
    I_ext = 0
    for time in tqdm(range(sim_steps)):

        # dynamics for Tsodyks-Markram model
        # x : membrane potential, u : vesicle release factor, φ : number of full vesicles
        # y : activity, T : total input, S : sensory input
        # w_jk : excitatory recurrent, z_jk : inhibitory rec, v_jl : exc sensory

        if neurons == 1 :
            if (time == 0): I_ext = -10
            if (time == 1000): I_ext = 10
            if (time == 2000): I_ext = -10

        dx, du_i, dφ_i, y, T = dynamics.tsodyks_markram(x, φ, u, w_jk, z_jk, I_ext)
       
        φ += dφ_i * delta_t
        u += du_i * delta_t
        full_vesicles_record[:, time] = φ
        vesic_release_record[:, time] = u

        x += dx * delta_t
        if time % (sim_steps//500) == 0:
            if np.isnan(x).any():
                print('\nNaN detected at time {}!\n'.format(time))
                sim_steps = time
                break
            if (x == membrane_pot_record[:, time-1]).all() and False:
                print('Fixpoint reached!')
                sim_steps = time
                break

        activity_record[:, time] = y
        membrane_pot_record[:, time] = x
        input_record[:, time] = T


    # End of ONE simulation

    '''
    fig_ac, ax_ac = myplot.activity(graph, time_plot, neurons_plot, activity_record, membrane_pot_record,
                                -sensory_inp_record, 0
                                , bars_time=bars_time)
    '''
    #is_1st_clique_active = (np.isclose(y[graph.clique_list[0]], 1, atol=0.01)).all()
    #is_2nd_clique_active = (np.isclose(y[graph.clique_list[1]], 1, atol=0.01)).all()

    clique_activity = [activity_record[i:i+clique_size].sum(axis=0) for i in range(0, neurons, clique_size)]
    clique_activity = np.array(clique_activity)
    cycler = myplot.rotating_cycler(n_clique)
    fig_cliq_ac, ax_cliq_ac = plt.subplots()
    ax_cliq_ac.set_prop_cycle(cycler)
    ax_cliq_ac.plot(clique_activity.T)
    # any clique with activity > 5 at any time (axis=1) is considered winning
    winning_clique = (clique_activity>5).any(axis=1).sum()
    how_many_winning_cliques.append(winning_clique)
    '''
    which_winning_clique = (clique_activity>5).any(axis=1).nonzero()[0]
    if which_winning_clique.size > 0:
        which_winning_clique = which_winning_clique[0]
    else:
        which_winning_clique = np.nan

    which_clique_won.append(which_winning_clique)

    final_state = membrane_pot_record[:, -1]
    middle_neurons = final_state[(final_state>-0.8) * (final_state<-.1)]
    how_many_middle_neurons = middle_neurons.shape[0]
    if how_many_middle_neurons == (n_clique - 1) * n_links:
        is_one_clique_winning = 1
    elif how_many_middle_neurons == neurons:
        is_one_clique_winning = 0
    else:
        print('Something is wrong, {} middle neurons'.format(how_many_middle_neurons))
        break

    final_states.append([middle_neurons.mean(), is_one_clique_winning])
    '''
# End of ALL simulations

# Plotting

# Setting up....


#final_times = 30000
#final_times = sim_steps

y_plot = activity_record[:, -final_times:]
input_pl = input_record[:, -final_times:]

x_plot = membrane_pot_record[:, -final_times:]
full_vesicles_plot = full_vesicles_record[:, -final_times:]
vesic_release_plot = vesic_release_record[:, -final_times:]
effective_weights_plot = vesic_release_plot * full_vesicles_plot

# Actually plotting stuff

list_of_plots = {}
list_of_data = {}
if neurons == 1 and not (vesic_release_record == 0).all():
    fig_stp, ax_stp = myplot.full_depletion(time_plot, full_vesicles_plot,
                                            vesic_release_plot)
    #plt.savefig('./notes/Poster/hendrik/images/double_depletion.pdf', dpi=300)
    list_of_plots['full_depletion'] = fig_stp
    list_of_data['full_depletion'] = {'time_plot' : time_plot, 
    'full_vesicles_plot' : full_vesicles_plot,
    'vesic_release_plot' : vesic_release_plot,}

save_figures = False

#fig_ac, ax_ac = myplot.activity(graph, time_plot, neurons_plot, y_plot)
#ax_ac.set_xlim([0,10])
#plt.savefig('./plots/double_activity.pdf', dpi=300)                              
#list_of_plots['activity'] = fig_ac

'''
final_states = np.array(final_states)
how_many_winning_cliques = np.array(how_many_winning_cliques)
which_clique_won = np.array(which_clique_won)
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(how_many_winning_cliques)
x_lim = ax_hist.get_xlim()
xlim_1 = (x_lim[1] - x_lim[0]) * 105/110 + x_lim[0]
ax_hist.set_xlim([-xlim_1*0.05, xlim_1 * 1.05])
ax_hist.set_xlabel('Winning cliques')
ax_hist.set_title('{} cliques, {} trials, {}to{}'.format(n_clique, trials_num, n_links, n_links))
path = 'plots/scaling_network/from{}to{}/'.format(n_links, n_links)
file_name = 'hist_winning_cliques{}x{}_c{}'.format(n_clique, trials_num, n_links)
#fig_hist.savefig(path + file_name)
list_of_plots['hist_winning_cliques'] = fig_hist
list_of_data['hist_winning_cliques'] = {'final_states' : final_states, 
    'how_many_winning_cliques' : how_many_winning_cliques,
    'which_clique_won' : which_clique_won}
'''
def save_stuff():
    version = 0
    max_attempts = 10
    for key, figure in list_of_plots.items():
        for attempt in range(max_attempts):
            try:
                dir_name = './log/{}/'.format(version)
                os.makedirs(dir_name)
                if version == max_attempts:
                    print('Too many logged files')
                else:
                    os.makedirs(dir_name + 'plots/')
                    figure.savefig(dir_name + 'plots/' + key, dpi=300)
                    os.makedirs(dir_name + 'data/')
                    np.savez_compressed(dir_name + 'data/' + key, **list_of_data[key])
                    os.makedirs(dir_name + 'code/')
                    file_name_list = ['clique.py', 'semantic.py', 'dynamics.py', 'plotting.py', 'networks.py']
                    for source_file in file_name_list:
                        shutil.copy('./'+source_file, dir_name + 'code/' + source_file)
                    break
            except OSError as error:
                version += 1
            else:
                print('This shouldn\'t happen')
                break



            #data_file = open('./log/data/sim_data{}.npz'.format(version), 'x')
            #np.savez_compressed('./log/data/sim_data{}.npz'.format(version), 
            #    **list_of_data['full_depletion'],
            #    **list_of_data['complete'],
            #    **list_of_data['hist_winning_cliques'],
            #    **list_of_data['not_a_key']
            #    )