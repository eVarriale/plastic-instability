#!/usr/bin/env python3

#import pdb                          # for debugging
import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from matplotlib import cm
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import networkx as nx               # for graphs
from cycler import cycler           # for managing plot colors
import dynamics


PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def rotating_cycler(n_clique):
    ''' returns a cycler that plt can use to assign colors '''
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    my_cycler = cycler('linestyle', ['-', '--', ':', '-.']) * cycler(color = colors)
    my_cycler = my_cycler[:n_clique]
    return my_cycler

def color_map(n_clique):
    ''' returns n_clique colors out of viridis color map. '''
    start = 0.
    stop = 1
    cm_subsection = np.linspace(start, stop, n_clique)
    colors = [cm.viridis(x) for x in cm_subsection]
    return colors


def geometric_shell(graph):
    ''' compute positions: cliques at regular angle intervals, and cliques as
    regular shapes'''
    radius = 2 * len(graph.clique_list)
    neuron_radius = 0.4 * len(graph.clique_list)
    npos = {}
    # Discard the extra angle since it matches 0 radians.
    theta_clique = -np.linspace(0, 1, len(graph.clique_list) + 1)[:-1] * 2 * np.pi 
    theta_clique += np.pi/2
    clique_center = radius *  np.column_stack([np.cos(theta_clique), np.sin(theta_clique)])
    for i, clique in enumerate(graph.clique_list):
        theta_neuron = -np.linspace(0, 1, len(clique)+1)[:-1] * 2 * np.pi
        clique_angle = theta_clique[i] - 2*np.pi/len(clique)/2
        turns = (i * (len(clique) - 2)) % len(clique)
        clique_angle -=  turns * 2 * np.pi/len(clique)
        pos = np.column_stack([np.cos(theta_neuron+clique_angle), np.sin(theta_neuron+clique_angle)])
        pos *= neuron_radius
        pos += clique_center[i]
        npos.update(zip(clique, pos))
    return npos

def neurons_to_clique(graph):
    neuron_to_clique = [0]*graph.number_of_nodes()
    for clique, clique_members in enumerate(graph.clique_list):
        for neuron in clique_members:
            neuron_to_clique[neuron] = clique
    return neuron_to_clique

def network(graph):
    ''' plots the graph using networkx '''
    fig_net, ax_net = plt.subplots()
    ax_net.axis('off')

    #pos = nx.circular_layout(graph)
    #pos = nx.shell_layout(graph)
    #pos = nx.spring_layout(graph)
    #pos = nx.kamada_kawai_layout(graph)
    pos = geometric_shell(graph)
    index_list = neurons_to_clique(graph)
    edgecolor = [COLORS[i%10] for i in index_list]
    color = "#{0:02x}{1:02x}{2:02x}".format(0,97,143)
    nx.draw_networkx_nodes(graph, pos=pos, node_color=color, 
                            edgecolors=edgecolor, linewidths=2)
    #nx.draw_networkx_nodes(graph, pos=pos)
    exc_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.]
    nx.draw_networkx_edges(graph, pos, edgelist=exc_edge, lw=10)

    #inh_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] < 0.]
    #nx.draw_networkx_edges(graph, pos, edgelist=inh_edge, style='dashed',
    #                       edge_color='b', alpha=0.5, lw=10)

    #clique = graph.clique_list[0]
    #clique_edge = [(a, b) for i, a in enumerate(clique) for b in clique[i+1:]]
    #nx.draw_networkx_edges(graph, pos, edgelist=clique_edge, edge_color='red', lw=10) #

    #nx.draw_networkx_labels(graph, pos)
    #labels = nx.get_edge_attributes(graph, 'weight')
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    plt.tight_layout()

    #plt.suptitle(graph.title)
    if graph.sparse:
        inhibition = 'Random inhibitory connections'
    else:
        inhibition = 'Full inhibitory background'
    #plt.title(inhibition)
    #plt.title(graph.title)
    fig_net.set_size_inches([6, 6])

    return fig_net, ax_net

def activity(graph, time_plot, neurons_plot, y_plot, other_plots=None, 
                save_figures=False, bars_time=None):

    fig, ax = plt.subplots()
    ax.set(ylabel='Activity y', xlabel='time (s)')
    my_kwargs = list(rotating_cycler(graph.n_c))
    #ax.set_prop_cycle(cycler)
    x_lim_index = min(4999, time_plot.shape[0]-1)
    ax.set_xlim(time_plot[-x_lim_index], time_plot[-1])

    #colors = color_map(graph.n_c)
    lines = []
    labels = []
    for i in range(neurons_plot):
        for clique, clique_members in enumerate(graph.clique_list):
            if i in clique_members: break

        label = clique if i == clique_members[0] else '_nolegend_'
        color = COLORS[clique%10]
        #color = colors[clique]
        line, = ax.plot(time_plot, y_plot[i].T, label=label, **my_kwargs[clique])
        lines.append(line)

        if other_plots is not None:
            for label_name, to_be_plot in other_plots.items():
                label = label_name if i == 0 else '_nolegend_'
                line, = ax.plot(time_plot, to_be_plot[i].T, label=label2, **my_kwargs[clique], linestyle='-.', alpha=0.5)
                lines.append(line)
    #legend1 = plt.legend(title='Cliques', loc=1, frameon=False)
    #ax.add_artist(legend1)
    
    ax.set_yticks([0, 1])
    plt.tight_layout()
    if save_figures:
        savefig_options = {'papertype' : 'a5', 'dpi' : 300}
        fig.savefig('{}x{}_s{}_r{}'.format(graph.n_c, graph.s_c, graph.sparse, gain_rule), **savefig_options)
        #plt.close(fig)
        #Y = 0
    return fig, ax

def full_depletion(time_plot, full_vesicles_inh_plot, vesic_release_inh_plot):

    fig_fulldep, ax_fulldep = plt.subplots()

    effective_weights_plot_inh = vesic_release_inh_plot * full_vesicles_inh_plot
    ax_fulldep.plot(time_plot, full_vesicles_inh_plot.T, label=r'$\varphi_i$')
    ax_fulldep.plot(time_plot, vesic_release_inh_plot.T, label='$u_i$')
    ax_fulldep.plot(time_plot, effective_weights_plot_inh.T, label=r'$\varphi_i \cdot u_i$')
    u_phi_max = effective_weights_plot_inh[:, 100:].max()
    ax_fulldep.set_yticks([0, 1, u_phi_max, dynamics.U_max])
    ax_fulldep.set_yticklabels(['0', '1', '{:2.1f}'.format(u_phi_max), '$U_{max}$'])
    ax_fulldep.legend(frameon=False, prop={'size': 15})

    ax_fulldep.set(ylim=[-0.02, dynamics.U_max + .02], xlim=[1.8, 3.8])

    #ax_fulldep.set_xticks([1.8, 2.3, 2.8, 3.3])
    #ax_fulldep.set_xticklabels(['0', '0.5', '1', '1.5'])
    plt.legend()
    plt.tight_layout()
    return fig_fulldep, ax_fulldep

def input_signal(graph, time_plot, neurons_plot, input_pl):
    fig_title_inp = 'Input of ' + graph.title
    fig_inp, ax_inp = plt.subplots()
    ax_inp.set(ylabel='Input', xlabel='time (s)', title=fig_title_inp)
    #my_cycler = rotating_cycler(graph.n_c)
    #ax_inp.set_prop_cycle(my_cycler)
    index_list = neurons_to_clique(graph)
    ax_inp.set_xlim(time_plot[-5000], time_plot[-1])
    for i in range(neurons_plot):
        label = i if i < graph.n_c else '_nolegend_'
        line, = ax_inp.plot(time_plot, input_pl[i].T, label=label, color=COLORS[index_list[i]%10])
    ax_inp.legend(frameon=False)
    return fig_inp, ax_inp