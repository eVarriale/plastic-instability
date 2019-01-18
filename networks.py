#!/usr/bin/env python3

import numpy as np                  # for math
import networkx as nx               # for graphs
import scipy.linalg as LA           # for linear algebra
#import pdb                          # for debugging
#pdb.set_trace()

def set_up_graph(weights, n_clique=0, clique_size=0, clique_list=[], sparse=False):
    graph = nx.Graph(weights)
    graph.n_c = n_clique
    graph.s_c = clique_size
    graph.sparse = sparse
    graph.clique_list = clique_list
    graph.title = '{} cliques with {} nodes'.format(graph.n_c, graph.s_c)
    return graph

def erdos_renyi(p_conn, neurons, w=2.4, z=-24):
    '''create a random excitatory network, and then complete the graph with inhibitory links'''

    adjacency_exc = np.triu(np.random.rand(neurons, neurons) < p_conn, k=1)
    # upper triangular matrix above diagonal
    #adjacency_exc += adjacency_exc.T # symmetrize
    adjacency_inh = np.triu(1 - adjacency_exc, k=1)

    dist_w = np.random.normal(w, w/60, adjacency_exc.shape)
    dist_z = np.random.normal(z, -z/60, adjacency_inh.shape)

    exc_weights = adjacency_exc * dist_w
    inh_weights = adjacency_inh * dist_z

    exc_weights += exc_weights.T
    inh_weights += inh_weights.T

    #np.fill_diagonal(weights, 0)
    graph = nx.Graph(exc_weights + inh_weights, weight=exc_weights + inh_weights)

    assert((np.diag(exc_weights) == 0).all())
    assert((np.diag(inh_weights) == 0).all())
    assert((exc_weights == exc_weights.T).all())
    assert((inh_weights == inh_weights.T).all())

    return exc_weights, inh_weights, graph

def single_neuron():
    exc_weights = np.array([[0.]])
    inh_weights = np.array([[0.]])
    n_clique = 1
    clique_size = 1
    clique_list = [[0]]
    weights = exc_weights+inh_weights
    graph = set_up_graph(weights, n_clique, clique_size, clique_list)
    return exc_weights, inh_weights, graph

def scaling_network(n_clique, clique_size=6, n_links=2, w_scale=1):

    w = 10 / (clique_size - 1)
    w_ext = w
    z = - 10 / clique_size

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

    dist_w = np.random.normal(w, w/60, adj_shape)
    dist_w = np.triu(dist_w, k=1) + np.triu(dist_w, k=1).T
    dist_w_ext = np.random.normal(w_ext, w_ext/60, adj_shape)
    dist_w_ext = np.triu(dist_w_ext, k=1) + np.triu(dist_w_ext, k=1).T
    dist_z = np.random.normal(z, -z/60, adj_shape)
    dist_z = np.triu(dist_z, k=1) + np.triu(dist_z, k=1).T

    weights =  w_scale *(adjacency_exc_intra * dist_w +
         adjacency_exc_extra * dist_w_ext + adjacency_inh * dist_z)
    assert((np.diag(weights)==0).all())
    assert((weights==weights.T).all())

    clique_list = [[i*clique_size + j for j in range(clique_size)] for i in range(n_clique)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse=False)

    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, graph

def hopfield_network(n_neu, n_patt):
    #capacity = int( 0.138 * n_neu)
    patterns = np.random.randint(2, size=(n_patt, n_neu)) * 2 - 1
    outer_products = np.array([np.outer(i, i) for i in patterns])
    weights = 1/n_patt * (outer_products.sum(axis=0) - n_patt * np.eye(n_neu))
    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    clique_list = [c for c in nx.find_cliques(nx.Graph(exc_weights)) if len(c)>2]
    n_c = len(clique_list)
    graph = set_up_graph(weights, clique_list=clique_list, n_clique=n_c)
    return exc_weights, inh_weights, graph, patterns
