# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:21:20 2021

@author: femiogundare
"""

import os
import re
import itertools
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import networkx as nx
from config import REACTOME_PATHWAY_PATH
from data_scripts.gmt_reader import GMTReader

pathway_names = 'ReactomePathways.txt'
pathway_genes = 'ReactomePathways.gmt'
relations_file = 'ReactomePathwaysRelation.txt'



def create_edges(node, n_levels):
    """
    Creates edges of a graph from a given node and number of levels. 
    
    Args:
        node : start point for creation of edges
        n_levels : number of edges to be created
    """
    edges_of_graph = []
    start_point = node    #start the creation of edges from this point
    for level in range(n_levels):
        current_point = node + '_copy' + str(level+1)    #create a new node
        edge = (start_point, current_point)    #the distance between the previous node and the new node is an edge
        start_point = current_point    #the new node becomes the current node from which subsequent nodes are created
        edges_of_graph.append(edge)
        
    return edges_of_graph
    

def add_edges(G, node, n_levels):
    """Adds edges to a pre-built graph."""
    G.add_edges_from(create_edges(node, n_levels))
    return G


def complete_network(G, n_leveles):
    #Create an ego network. An ego network is a special type of network consisting of one central node and all other nodes directly connected to it.
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    #Obtain the terminal nodes of the ego graph. Terminal nodes in this case are nodes with an out degree of zero.
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]

    for node in terminal_nodes:
        # Find the number of shortest paths there is between the source node and the current node
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        #If the number of the shortest paths is less than of equal to n_levels, add diff edges to the graph
        #where diff is n_leveles - distance + 1
        if distance <= n_leveles:    
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph



def get_nodes_at_level(net, distance):
    # Get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    
    # Remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
        
    return layers



class Reactome:
    """
    Reactome class.
    
    Basically loads the datasets containing the names of the pathways, names of the genes involved in the pathways,
    and the pathways' relations.
    """
    def __init__(self):
        self.pathway_names = self.load_pathway_names()
        self.pathway_genes = self.load_pathway_genes()
        self.hierarchy = self.load_hierarchy()
        
    def load_pathway_names(self):
        filename = os.path.join(REACTOME_PATHWAY_PATH, pathway_names)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'specie']
        return df
    
    def load_pathway_genes(self):
        filename = os.path.join(REACTOME_PATHWAY_PATH, pathway_genes)
        gmt_reader = GMTReader()
        df = gmt_reader.load_data(filename, pathway_col=1, genes_col=3)
        return df
    
    def load_hierarchy(self):
        filename = os.path.join(REACTOME_PATHWAY_PATH, relations_file)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df
    
    
class ReactomeNetwork:
    """
    Reactome network class.
    
    Basically builds a network using the reactome datasets.
    """
    def __init__(self):
        self.reactome = Reactome()
        self.netx = self.get_reactome_networkx()
        
    def get_terminal_nodes(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes
    
    def get_root_nodes(self):
        root_nodes = get_nodes_at_level(self.netx, distance=1)
        return root_nodes
    
    #Get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        #Filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'
        
        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net
    
    def info(self):
        return nx.info(self.netx)
    
    def get_tree(self):
        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')
        return G
    
    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G
    
    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G
    
    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]
            
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers
    
    
def get_map_from_layer(layer_dict):
    """
    Args:
        layer_dict : dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}
                                                
    Returns:
        dataframe map of layer (index = genes, columns = pathways, , values = 1 if connected; 0 else)
    """
    pathways = layer_dict.keys()
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    df = pd.DataFrame(index=pathways, columns=genes)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T
    
    
if __name__ == '__main__':
    reactome = Reactome()
    pathway_names_df = reactome.pathway_names
    pathway_genes_df = reactome.pathway_genes
    hierarchy_df = reactome.hierarchy
    print('Pathway names\n', pathway_names_df.head())
    print('\nPathway genes\n', pathway_genes_df.head())
    print('\nHierarchy\n', hierarchy_df.head())
    
    reactome_network = ReactomeNetwork()
    print(reactome_network.info())
      
    print('Number of root nodes {}, Number of terminal nodes {}'.format(len(reactome_network.get_root_nodes()), len(reactome_network.get_terminal_nodes())))
    print(nx.info(reactome_network.get_completed_tree(n_levels=5)))
    print(nx.info(reactome_network.get_completed_network(n_levels=5)))
    layers = reactome_network.get_layers(n_levels=3)
    print(len(layers))
    
    for i, layer in enumerate(layers[::-1]):
        mapp = get_map_from_layer(layer)
        if i == 0:
            genes = list(mapp.index)[0:5]
        filter_df = pd.DataFrame(index=genes)
        all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        genes = list(mapp.columns)
        print(all.shape)