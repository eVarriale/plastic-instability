#!/usr/bin/env python3

import importlib                    # for reimporting libraries
import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from tqdm import tqdm               # for progress bar
import os                           # for saving stuff
import shutil                       # for copying source code
import networkx as nx
import itertools                    # for product of iterators
#import pdb                          # for debugging
#pdb.set_trace()

#plt.style.use('seaborn-poster')
#plt.style.use('seaborn-talk')
plt.style.use('seaborn-ticks')

import networks as net
import dynamics as dyn
import plotting as myplot

plt.ion()

def reload_mod():
    importlib.reload(net)
    importlib.reload(dyn)
    importlib.reload(myplot)

def one_run(x_0=None, φ_0=None, u_0=None, model="TM"):
    '''
    Runs a model with random initial conditions.

     Tsodyks-Markram model:
     x : membrane potential,
     u : vesicle release factor,
     φ : number of full vesicles
     y : activity,
     T : total input,
     S : sensory input
     w_jk : excitatory recurrent weights,
     z_jk : inhibitory recurrent weights,
     v_jl : excitatory sensory weights
    '''
    if x_0 is None:
        x_0 = np.random.uniform(-1, 1, neurons)
    if φ_0 is None:
        if model != "NP":
            φ_0 = 1 * np.random.normal(1, 0.1, neurons)
        elif model == "NP":
            φ_0 = np.ones(1)
    if u_0 is None:
        if model != "NP":
            u_0 = 1 * np.random.normal(1, 0.1, neurons)
        elif model == "NP":
            u_0 = np.ones(1)

    x = x_0.copy()
    u = u_0.copy()
    φ = φ_0.copy()

    neurons_over_time = (neurons, sim_steps)
    y_rec = np.zeros(neurons_over_time)
    x_rec = np.zeros(neurons_over_time)
    input_rec = np.zeros(neurons_over_time)
    φ_rec = np.zeros(neurons_over_time)
    u_rec = np.zeros(neurons_over_time)

    if model == "TM":
        flow = dyn.tsodyks_markram
    elif model == "FD":
        flow = dyn.full_depletion
    elif model == "NP":
        flow = dyn.simple_flow

    I_ext = 0
    sim_iterator = tqdm(range(sim_steps))
    for time in sim_iterator:

        if neurons == 1 :
            if (time == 0): I_ext = -10
            if (time == 1000): I_ext = 10
            if (time == 2000): I_ext = -10

        dx, du_i, dφ_i, y, T = flow(x, φ, u, w_jk, z_jk, I_ext, dyn.tanhyp)
        #dx, du_i, dφ_i, y, T = dyn.full_depletion(x, φ, u, w_jk, z_jk)
        φ += dφ_i * delta_t
        u += du_i * delta_t
        φ_rec[:, time] = φ
        u_rec[:, time] = u

        x += dx * delta_t
        if time % (sim_steps//500) == 0:
            if np.isnan(x).any():
                print('\nNaN detected at time {}!\n'.format(time))
                #sim_steps = time
                sim_iterator.close()
                break
            if (abs(dx) < 1e-6).all():
                #print('\nFixpoint reached!\n')
                #sim_steps = time
                fixpoint_reached[trial_tuple] = time
                is_fixpoint[trial_tuple] = True
                sim_iterator.close()
                break

        y_rec[:, time] = y
        x_rec[:, time] = x
        input_rec[:, time] = T

    return y_rec, x_rec, input_rec, φ_rec, u_rec

# Global variables

sim_steps = 2 * 60 * 1000 # integration steps
delta_t = 1 # integration time step in milliseconds

w_mean = 1

n_clique = 5
clique_size = 6 # >2 !, otherwise cliques don't make sense
n_links = 1

seed = 42
np.random.seed(seed)
model = "TM" # Tsodyks-Markram model
#model = "FD" # Full depletion model
#model = "NP" # No STSP model

# Initialize stuff

#w_jk, z_jk, graph = net.erdos_renyi(0.3, 30)
#w_jk, z_jk, graph = net.single_neuron()
#w_jk, z_jk, graph = net.scaling_network(n_clique, clique_size=clique_size)
final_times = np.minimum(sim_steps, 20000)
time_plot = np.arange(sim_steps-final_times, sim_steps)/int(1000/delta_t)
# time in seconds

how_many_winning_cliques = []
final_states = []
which_clique_won = []

fixpoint_num = []
correct_pattern = []

saved_patterns = np.arange(2, 15)
diff_net = 100
overlap = []
for num_patterns in tqdm(saved_patterns):

    u_m_trials = np.array([5])
    #n_clique_trials = np.arange(2, 13)
    n_pattern_trials = np.full(diff_net, num_patterns) # shape, fill value
    #n_clique_trials = np.ones(100)

    trials_num = (len(n_pattern_trials), len(u_m_trials))
    iterator = itertools.product(n_pattern_trials, u_m_trials)

    is_fixpoint = np.zeros(trials_num, dtype='bool')
    fixpoint_reached = np.full(trials_num, sim_steps)
    patterns_found = np.zeros(trials_num, dtype='bool')
    which_pattern_found = []
    avg_overlap = 0
    for trial, parameters in enumerate(tqdm(iterator)):
        trial_tuple = (trial//len(u_m_trials), trial%len(u_m_trials) )
        n_pattern, dyn.U_m = parameters
        #dyn.U_m = u_m_trials[trial]
        #n_clique = n_clique_trials[trial]
        #w_jk, z_jk, graph = net.scaling_network(n_clique)
        w_jk, z_jk, graph, patterns = net.hopfield_network(100, n_pattern)
        weights = w_jk + z_jk
        neurons = graph.number_of_nodes()

        neurons_plot = neurons

        # Main simulation

        I_ext = 0
        y_rec, x_rec, input_rec, φ_rec, u_rec = one_run(x_0=patterns[0].astype('float'), model=model)
        # End of ONE simulation

        if neurons == 1:
            y_plot = y_plot = y_rec[:, -final_times:]
            fig_ac, ax_ac = myplot.activity(graph, time_plot, y_plot)

        if is_fixpoint[trial_tuple]:
            fix_time = fixpoint_reached[trial_tuple]
            time_plot = np.arange(fix_time)/int(1000/delta_t)
            similarity = np.dot(patterns, y_rec[:, :fix_time])/neurons
            plus_pattern = np.isclose(similarity[:,-1], 1, rtol=1e-01)
            minus_pattern = np.isclose(similarity[:,-1], -1, rtol=1e-01)
            patterns_found[trial_tuple] = ((plus_pattern + minus_pattern).sum())
            if patterns_found[trial_tuple] == 1:
                if plus_pattern.nonzero()[0].size >0:
                    the_pattern = plus_pattern.nonzero()[0][0] + 1
                else:
                    the_pattern = -minus_pattern.nonzero()[0][0] - 1
                which_pattern_found.append(the_pattern)
            elif patterns_found[trial_tuple] > 1:
                which_pattern_found.append(100 + patterns_found[trial_tuple])
            avg_overlap += np.dot(patterns[0], y_rec[:, fix_time-1])/neurons
            #fig_ac, ax_ac = plt.subplots()
            #ax_ac.plot(time_plot, y_rec[:, :fix_time].T)
            #ax_ac.set_title('Activity')
            #print(pattern_found)
            #fig_sim, ax_sim = plt.subplots()
            #ax_sim.plot(time_plot, similarity.T)

        if (not is_fixpoint[trial_tuple] and neurons > 1):
            fix_time = fixpoint_reached[trial_tuple]
            #fix_time = final_times
            time_plot = np.arange(fix_time)/int(1000/delta_t)
            if True:
                similarity = np.dot(patterns, y_rec[:, :fix_time])/neurons
                fig_sim, ax_sim = plt.subplots()
                ax_sim.plot(time_plot, similarity.T)
                ax_sim.set_title('Pattern similarity')
            else:
                clique_list = [c for c in nx.find_cliques(nx.Graph(w_jk)) if len(c)>2]
                clique_names = [str(c) for c in clique_list]
                clique_activity = [y_rec[c].sum(axis=0) for c in clique_list]
                clique_activity = np.array(clique_activity) / clique_size
                cycler = myplot.rotating_cycler(len(clique_list))
                fig_cliq_ac, ax_cliq_ac = plt.subplots()
                ax_cliq_ac.set_prop_cycle(cycler)
                ax_cliq_ac.plot(time_plot, clique_activity[:, :fix_time].T)
                plt.legend(ax_cliq_ac.get_lines(), clique_names)
                fig_cliq_ac.suptitle('Average clique activity')
                ax_cliq_ac.set_title('T_f={}, T_u={}, U={}, n_c={}'.format(dyn.T_φ, dyn.T_u, dyn.U_m, n_clique))
                ax_cliq_ac.set_xlim([time_plot[-5000], time_plot[-1]])
                #fig_cliq_ac.savefig('./plots/not_fix_{}.pdf'.format(trial), dpi=300)
                #plt.close(fig_cliq_ac)

                # clique_phi = [φ_rec[c].sum(axis=0) for c in clique_list]
                # clique_phi=np.array(clique_phi)/6
                # clique_u=[u_rec[c].sum(axis=0) for c in clique_list]
                # clique_u=np.array(clique_u)/6
                # fig_stp, ax_stp = myplot.full_depletion(time_plot, clique_phi, clique_u)
                # fig_stp.suptitle('Clique synaptic efficacy')
                # ax_stp.set_title('T_f={}, T_u={}'.format(dyn.T_φ, dyn.T_u))
                # ax_stp.set_xlim([-.5, 2])

                # any clique with activity > 5 at any time (axis=1) is considered winning
                winning_clique = (clique_activity>0.7).any(axis=1).sum()
                how_many_winning_cliques.append(winning_clique)


        # which_winning_clique = (clique_activity>5).any(axis=1).nonzero()[0]
        # if which_winning_clique.size > 0:
        #     which_winning_clique = which_winning_clique[0]
        # else:
        #     which_winning_clique = np.nan
        #
        # which_clique_won.append(which_winning_clique)
        #
        # final_state = x_rec[:, -1]
        # middle_neurons = final_state[(final_state>-0.8) * (final_state<-.1)]
        # how_many_middle_neurons = middle_neurons.shape[0]
        # if how_many_middle_neurons == (n_clique - 1) * n_links:
        #     is_one_clique_winning = 1
        # elif how_many_middle_neurons == neurons:
        #     is_one_clique_winning = 0
        # else:
        #     print('Something is wrong, {} middle neurons'.format(how_many_middle_neurons))
        #     break
        #
        # final_states.append([middle_neurons.mean(), is_one_clique_winning])

        #which_pattern_found = np.array(which_pattern_found)
        #fig_pat, ax_pat = plt.subplots()
        #ax_pat.hist(which_pattern_found)

    fixpoint_num.append(is_fixpoint.mean())
    correct_pattern.append(patterns_found.mean())
    overlap.append(avg_overlap/diff_net)

# End of ALL simulations

fixpoint_num = np.array(fixpoint_num)
correct_pattern = np.array(correct_pattern)
overlap = np.array(overlap)
np.savez_compressed('./log/hopfield_np.npz', saved_patterns=saved_patterns,
                    np_found=correct_pattern, sigmoidal=fixpoint_num)

# Plotting
# Setting up....


#final_times = 30000
#final_times = sim_steps

y_plot = y_rec[:, -final_times:]
input_pl = input_rec[:, -final_times:]

x_plot = x_rec[:, -final_times:]
full_vesicles_plot = φ_rec[:, -final_times:]
vesic_release_plot = u_rec[:, -final_times:]
effective_weights_plot = vesic_release_plot * full_vesicles_plot

# Actually plotting stuff

list_of_plots = {}
list_of_data = {}
if neurons == 1 and not (u_rec == 0).all():
    fig_stp, ax_stp = myplot.full_depletion(time_plot, full_vesicles_plot,
                                            vesic_release_plot)
    #plt.savefig('./notes/Poster/hendrik/images/double_depletion.pdf', dpi=300)
    list_of_plots['full_depletion'] = fig_stp
    list_of_data['full_depletion'] = {'time_plot' : time_plot,
    'full_vesicles_plot' : full_vesicles_plot,
    'vesic_release_plot' : vesic_release_plot,}

save_figures = False

#fig_ac, ax_ac = myplot.activity(graph, time_plot, y_plot)
#ax_ac.set_xlim([0,10])
#plt.savefig('./plots/double_activity.pdf', dpi=300)
#list_of_plots['activity'] = fig_ac


# final_states = np.array(final_states)
# how_many_winning_cliques = np.array(how_many_winning_cliques)
# which_clique_won = np.array(which_clique_won)
# fig_hist, ax_hist = plt.subplots()
# ax_hist.hist(how_many_winning_cliques)
# x_lim = ax_hist.get_xlim()
# xlim_1 = (x_lim[1] - x_lim[0]) * 105/110 + x_lim[0]
# ax_hist.set_xlim([-xlim_1*0.05, xlim_1 * 1.05])
# ax_hist.set_xlabel('Winning cliques')
# ax_hist.set_title('{} cliques, {} trials, {}to{}'.format(n_clique, trials_num, n_links, n_links))
# path = 'plots/scaling_network/from{}to{}/'.format(n_links, n_links)
# file_name = 'hist_winning_cliques{}x{}_c{}'.format(n_clique, trials_num, n_links)
# #fig_hist.savefig(path + file_name)
# list_of_plots['hist_winning_cliques'] = fig_hist
# list_of_data['hist_winning_cliques'] = {'final_states' : final_states,
#     'how_many_winning_cliques' : how_many_winning_cliques,
#     'which_clique_won' : which_clique_won}


# np_found = np.array([100, 85, 73, 63, 74, 60, 58, 37, 34, 8, 3, 1, 0, 0])
# np_fix = np.full(14, 100)
# tm_found = np.array([100, 82, 73, 41, 17,  4,  0,  0, 0,  0, 0, 0, 0, 0])
# tm_fix = np.array([100, 100, 99, 96, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
fig_fx, ax_fx = plt.subplots()
ax_fx.plot(saved_patterns, fixpoint_num/diff_net, label='# fixpoints without STSP')
ax_fx.plot(saved_patterns, correct_pattern/diff_net, label='# found patterns without STSP')
ax_fx.plot(saved_patterns, overlap, label='# final overlap without STSP')
ax_fx.legend()
# ax_fx.plot(saved_patterns, tm_fix, label='# fixpoints with TM')
# ax_fx.plot(saved_patterns, tm_found, label='# found patterns with TM')
# ax_fx.legend()
# ax_fx.set_title("100 different networks with 100 nodes")
# ax_fx.set(xlabel="# of saved patterns")
hea1der = ("###################################\n" +
         "Short term plasticity {} on scaling networks\n".format(model) +
         "Dynamics \n" +
         "U_m = {}, T_u = {}, T_φ = {}, ".format(dyn.U_m, dyn.T_u, dyn.T_φ) +
         "α = {}, T_x = {}, gain = {}, ".format(dyn.α, dyn.T_x, dyn.gain) +
         "sim_steps = {}\n".format(sim_steps) +
         "Network \n" + "n_c = {}, s_c = {}, ".format(n_clique, clique_size) +
         "w = w_e = {}, z = {} \n".format(w_jk.max(), z_jk.min()) +
         "###################################")



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
