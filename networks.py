#!/usr/bin/env python3

import numpy as np                  # for math
import networkx as nx               # for graphs
import scipy.linalg as LA           # for linear algebra
#import pdb                          # for debugging
#pdb.set_trace()


def set_up_graph(weights, n_clique, clique_size, clique_list, sparse=False):
    graph = nx.Graph(weights)
    graph.n_c = n_clique
    graph.s_c = clique_size
    graph.sparse = sparse
    graph.clique_list = clique_list
    graph.title = '{} cliques with {} nodes'.format(graph.n_c, graph.s_c)
    return graph

def erdos_renyi(p_conn, neurons, w_mean):
    '''create a random excitatory network, and then complete the graph with inhibitory links'''
    w = 2.4
    z = -24

    adjacency_exc = np.triu(np.random.rand(neurons, neurons) < p_conn, k=1)
    # upper triangular matrix above diagonal
    #adjacency_exc += adjacency_exc.T # symmetrize 
    adjacency_inh = np.triu(1 - adjacency_exc, k=1)
    
    weights = adjacency_exc - adjacency_inh
    weights *= np.random.normal(w_mean, 0.1*w_mean, weights.shape)
    weights += weights.T
    np.fill_diagonal(weights, 0)
    graph = nx.Graph(weights, weight=weights)
    return weights, graph

def single_neuron():
    exc_weights = np.array([[0.]])
    inh_weights = np.array([[0.]])
    n_clique = 1
    clique_size = 1
    clique_list = [0]
    weights = exc_weights+inh_weights
    graph = set_up_graph(weights, n_clique, clique_size, clique_list)
    return exc_weights, inh_weights, graph

def scaling_network(n_clique, clique_size=6, n_links=2, w_scale=1):

    w = 10 / (clique_size - 1)
    w_ext = w
    z = - 80 / clique_size

    neurons = n_clique * clique_size
    adj_shape = (neurons, neurons)
    cliques_adj = [1 - np.eye(clique_size)] * n_clique
    adjacency_exc_intra = LA.block_diag(*cliques_adj)
    adjacency_exc_extra = np.zeros(adj_shape)
    for i in range(0, neurons, clique_size):
        for j in range(i+clique_size, neurons, clique_size):
            rand_from = np.random.choice(clique_size, size=n_links, replace=False)
            rand_to = np.random.choice(clique_size, size=n_links, replace=False)
            adjacency_exc_extra[i + rand_from, j + rand_to] = 1

    for i in range(0, neurons, clique_size):
        for j in range(i+clique_size, neurons, clique_size):
                conn = adjacency_exc_extra[i:i+clique_size, j:j+clique_size].sum()
                assert(conn == n_links)
    adjacency_exc_extra += adjacency_exc_extra.T
    adjacency_inh = 1 - np.eye(neurons) - adjacency_exc_intra - adjacency_exc_extra

    norm_w = np.random.normal(1, w/60, adj_shape)
    norm_w_ext = np.random.normal(1, w_ext/60, adj_shape)
    norm_z = np.random.normal(1, -z/60, adj_shape)
    weights =  w_scale *(w * adjacency_exc_intra * norm_w + 
         w_ext * adjacency_exc_extra * norm_w_ext + z * adjacency_inh * norm_z)
    assert((np.diag(weights)==0).all())
    assert((weights==weights.T).all())

    clique_list = [[i*clique_size + j for j in range(clique_size)] for i in range(n_clique)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse=False)
    
    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, graph
