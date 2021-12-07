# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:12:11 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import itertools
import networkx as nx
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.offline import plot
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from visualization_utils import get_reactome_pathway_names

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PATHWAY_PATH, PLOTS_PATH

extracted_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_extracted')
saving_dir = os.path.join(PLOTS_PATH, 'figure3')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


def get_number_of_best_nodes(coefs):
    ind_source = (coefs - coefs.median()).abs() > 2. * coefs.std()
    num = min([10, int(sum(ind_source))])
    return num


def get_first_layer(node_weights, number_of_best_nodes, col_name='coef', include_others=True):
    feature_weights = node_weights[0].copy()
    gene_weights = node_weights[1].copy()
    
    feature_weights = feature_weights[[col_name]]
    gene_weights = gene_weights[[col_name]]
    
    if number_of_best_nodes == 'auto':
        coefs = gene_weights[col_name].sort_values()
        num = get_number_of_best_nodes(coefs)
        top_genes = list(gene_weights.nlargest(num, col_name).index)
    else:
        top_genes = list(gene_weights.nlargest(number_of_best_nodes, col_name).index)
        
    print('Top genes: {}'.format(top_genes))
    genes = gene_weights.loc[top_genes]
    genes[col_name] = np.log(1. + genes[col_name].abs())
    
    df = feature_weights
    
    if include_others:
        df = df.reset_index()
        df.columns = ['target', 'source', 'value']
        df['target'] = df['target'].map(lambda x: x if x in top_genes else 'others1')
        df = df.groupby(['source', 'target']).sum()
        df = df.reset_index()
        df.columns = ['source', 'target', 'value']
    else:
        df = feature_weights.loc[top_genes]
        df = df.reset_index()
        df.columns = ['target', 'source', 'value']
        
    df['direction'] = df['value'] >= 0.
    df['value'] = abs(df['value'])
    
    df['source'] = df['source'].replace('important_mutations', 'Mutation')
    df['source'] = df['source'].replace('cnv', 'Copy-number Variation')
    df['source'] = df['source'].replace('cnv_amplification', 'Amplification')
    df['source'] = df['source'].replace('cnv_deletion', 'Deletion')
    
    df['layer'] = 0
    
    # Perform normalization by gene groups
    df['value'] = df['value'] / df.groupby('target')['value'].transform(np.sum)
    df = df[df.value > 0.0]
    
    # Combine feature weights and gene weights
    df = pd.merge(df, genes, left_on='target', right_index=True, how='left')
    df.coef.fillna(10.0, inplace=True)
    df.value = df.value * df.coef * 150.
    
    return df


def get_first_layer_df(nlargest):
    features_weights = pd.read_csv(os.path.join(extracted_dir, 'gradient_importance_0.csv'), index_col=[0, 1])
    features_weights['layer'] = 0
    nodes_per_layer0 = features_weights[['layer']]
    features_weights = features_weights[['coef']]
    
    all_weights = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
    genes_weights = all_weights[all_weights.layer == 1]
    nodes_per_layer1 = genes_weights[['layer']]
    genes_weights = genes_weights[['coef_combined']]
    genes_weights.columns = ['coef']
    
    node_weights = [features_weights, genes_weights]
    
    df = get_first_layer(node_weights, number_of_best_nodes=nlargest[0], col_name='coef', include_others=True)
    first_layer_df = df[['source', 'target', 'value', 'layer']]
    
    return first_layer_df


def get_high_nodes(adjusted_node_importances, nlargest, column):
    layers = np.sort(adjusted_node_importances.layer.unique())
    high_nodes = []
    
    for i, layer in enumerate(layers):
        if type(nlargest) == list:
            n = nlargest[i]
        else:
            n = nlargest
        
        nodes = list(adjusted_node_importances[adjusted_node_importances.layer == layer].nlargest(n, columns=column).index)
        high_nodes.extend(nodes)
        
    return high_nodes


def filter_nodes(adjusted_node_importances, high_nodes, add_others=True):
    high_nodes_df = adjusted_node_importances[adjusted_node_importances.index.isin(high_nodes)].copy()
    if add_others:
        layers = list(adjusted_node_importances.layer.unique())
        names = ['others{}'.format(layer) for layer in layers]
        names = names + ['root']
        layers.append(np.max(layers) + 1)
        data = {'index': names, 'layer': layers}
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('index')
        high_nodes_df = high_nodes_df.append(df)
        
    return high_nodes_df


def get_links():
    """
    Returns a dataframe with all the connections in the model (except first layer).
    """
    links = []
    
    for layer_index in range(1, 7):
        link = pd.read_csv(os.path.join(extracted_dir, 'link_weights_{}.csv'.format(layer_index)), index_col=0)
        link.index.name = 'source'
        link.reset_index(inplace=True)
        link_unpivoted = pd.melt(link, id_vars=['source'], var_name='target', value_name='value')
        link_unpivoted['layer'] = layer_index
        link_unpivoted = link_unpivoted[link_unpivoted.value != 0.]
        link_unpivoted = link_unpivoted.drop_duplicates(subset=['source', 'target'])
        links.append(link_unpivoted)
    all_links_df = pd.concat(links, axis=0)
    
    return all_links_df


def get_MDM4_nodes(links_df):
    net = nx.from_pandas_edgelist(links_df, 'target', 'source', create_using=nx.DiGraph())
    net.name = 'Reactome'
    # Add the root node
    roots = [node for node, degree in net.in_degree() if degree == 0]
    root_node = 'root'
    edges = [(root_node, node) for node in roots]
    net.add_edges_from(edges)
    
    # Convert to tree
    tree = nx.bfs_tree(net, 'root')
    
    traces = list(nx.all_simple_paths(tree, 'root', 'MDM4'))
    all_mdm4_nodes = []
    for trace in traces:
        print('--'.join(trace))
        all_mdm4_nodes.extend(trace)
    
    nodes = np.unique(all_mdm4_nodes)
    
    return nodes


def filter_connections(df, high_nodes, add_unk=False):
    high_nodes.append('root')
    
    def apply_others(row):
        if not row['source'] in high_nodes:
            row['source'] = 'others' + str(row['layer'])
        if not row['target'] in high_nodes:
            row['target'] = 'others' + str(row['layer'] + 1)
        return row
    
    layers_ids = np.sort(df.layer.unique())
    layer_dfs = []
    
    for i, layer_id in enumerate(layers_ids):
        layer_df = df[df.layer == layer_id].copy()
        ind_1 = layer_df.source.isin(high_nodes)
        ind_2 = layer_df.target.isin(high_nodes)
        
        if add_unk:
            layer_df = layer_df[ind_1 | ind_2]
            layer_df = layer_df.apply(apply_others, axis=1)
        else:
            layer_df = layer_df[ind_1 & ind_2]

        layer_df = layer_df.groupby(['source', 'target']).agg({'value': 'sum', 'layer': 'min'})
        layer_dfs.append(layer_df)
        
    output = pd.concat(layer_dfs)
    
    return output


def get_node_colors(nodes, remove_others=True):
    color_idxs = np.linspace(1, 0, len(nodes))
    cmp = plt.cm.Reds
    node_colors = {}
    
    for idx, node in zip(color_idxs, nodes):
        if 'other' in node:
            if remove_others:
                c = (255, 255, 255, 0.0)
            else:
                c = (232, 232, 232, 0.5)
        else:
            colors = list(cmp(idx))
            colors = [int(255 * c) for c in colors]
            # Set alpha
            colors[-1] = 0.7
            c = colors
            
        node_colors[node] = c
    
    return node_colors


def get_node_colors_ordered(high_nodes_df, col_name, remove_others=True):
    node_colors = {}
    layers = high_nodes_df.layer.unique()
    
    for layer in layers:
        nodes_ordered = high_nodes_df[high_nodes_df.layer == layer].sort_values(col_name, ascending=False).index
        node_colors.update(get_node_colors(nodes_ordered, remove_others))
        
    return node_colors


def get_edge_colors(df, node_colors_dict, remove_others=True):
    edge_colors = []
    
    for _, row in df.iterrows():
        if ('others' in row['source']) or ('others' in row['target']):
            if remove_others:
                edge_colors.append('rgba(255, 255, 255, 0.0)')
            else:
                edge_colors.append('rgba(192, 192, 192, 0.2)')
        else:
            base_color = [color for color in node_colors_dict[row['source']]]
            # Set alpha
            base_color[-1] = 0.2
            base_color = 'rgba{}'.format(tuple(base_color))
            edge_colors.append(base_color)
            
    return edge_colors


def encode_nodes(df):
    source = df['source']
    target = df['target']
    all_nodes = list(np.unique(np.concatenate([source, target])))
    n_nodes = len(all_nodes)
    node_codes = range(n_nodes)
    df_encoded = df.replace(all_nodes, node_codes)
    return df_encoded, all_nodes, node_codes


def get_x_y(df_encoded, node_layers):
    source_weights = df_encoded.groupby(by='source')['value'].sum()
    target_weights = df_encoded.groupby(by='target')['value'].sum()
    
    node_weights = pd.concat([source_weights, target_weights])
    node_weights = node_weights.to_frame()
    node_weights = node_weights.groupby(node_weights.index).max()
    
    node_weights = node_weights.join(node_layers)
    idxs = node_weights.index.str.contains('others')
    others_values = node_weights.loc[idxs, 'value']
    node_weights.loc[idxs, 'value'] = 0.
    node_weights.sort_values(by=['layer', 'value'], ascending=False, inplace=True)
    node_weights.loc[others_values.index, 'value'] = others_values
    print(node_layers['layer'].unique())
    n_layers = len(node_layers['layer'].unique())
    print(n_layers)
    
    node_weights['x'] = (node_weights['layer'] - 2) * 0.1 + 0.16
    ind = node_weights.layer == 0
    node_weights.loc[ind, 'x'] = 0.01
    ind = node_weights.layer == 1
    node_weights.loc[ind, 'x'] = 0.08
    ind = node_weights.layer == 2
    node_weights.loc[ind, 'x'] = 0.14
    
    xs = np.linspace(0.14, 1, 6, endpoint=False)
    for i, x in enumerate(xs[1:]):
        ind = node_weights.layer == i + 3
        node_weights.loc[ind, 'x'] = x
        
    dd = node_weights.groupby('layer')['value'].transform(pd.Series.sum)
    node_weights['layer_weight'] = dd
    node_weights['y'] = node_weights.groupby('layer')['value'].transform(pd.Series.cumsum)
    node_weights['y'] = (node_weights['y'] - 0.5 * node_weights['value']) / (1.5 * node_weights['layer_weight'])
    
    # Root node
    ind = node_weights.layer==7
    node_weights.loc[ind, 'y'] = 0.33
    
    node_weights.sort_index(inplace=True)
    
    return node_weights['x'], node_weights['y']


def get_formated_network(links, high_nodes_df, col_name, remove_others):
    node_colors = get_node_colors_ordered(high_nodes_df, col_name, remove_others)
    node_colors['amplification'] = (224, 123, 57, 0.7)  #Amplification
    node_colors['deletion'] = (1, 55, 148, 0.7)  #Deletion
    node_colors['mutation'] = (105, 189, 210, 0.7)  #Mutation
    
    links['color'] = get_edge_colors(links, node_colors, remove_others)
    
    # Get node colors
    for key, value in node_colors.items():
        node_colors[key] = 'rgba{}'.format(tuple(value))
        
    # Remove links with no values
    links = links.dropna(subset=['value'], axis=0)
    
    links_encoded_df, all_nodes, node_codes = encode_nodes(links)
    node_layers_df = high_nodes_df[['layer']]
    
    # Remove self connection
    idxs = links_encoded_df.source == links_encoded_df.target
    links_encoded_df = links_encoded_df[~idxs]
    
    # Ensure there are positive values for all edges
    links_encoded_df.value = links_encoded_df.value.abs()
    
    x, y = get_x_y(links, node_layers_df)
    
    def pathways_short_names(all_nodes):
        df = pd.read_excel(os.path.join(PATHWAY_PATH, 'pathways_short_names.xlsx'), index_col=0)
        mapping_dict = {}
        for k, v in zip(df['Full name'].values, df['Short name (Eli)'].values):
            mapping_dict[k] = str(v)
            
        nodes_short_names = []
        for node in all_nodes:
            short_name = node
            if node in mapping_dict.keys() and not mapping_dict[node] == 'NaN' and not mapping_dict[node] == 'nan':
                short_name = mapping_dict[node]
            
            if 'others' in node:
                short_name = 'Residual'
            if 'root' in node:
                short_name = 'Outcome'
            
            nodes_short_names.append(short_name)
        
        return nodes_short_names
    
    node_colors_list = []
    for node in all_nodes:
        node_colors_list.append(node_colors[node])
    
    all_nodes_short_names = pathways_short_names(all_nodes)
    
    data = np.column_stack((node_codes, node_colors_list, all_nodes_short_names))
    nodes_df = pd.DataFrame(data, columns=['code', 'color', 'short_name'], index=all_nodes)
    nodes_df = nodes_df.join(x, how='left')
    nodes_df = nodes_df.join(y, how='left')
    
    return links_encoded_df, nodes_df


def get_data_trace(links, node_df, height, width, fontsize=6):
    all_nodes = node_df.short_name.values
    node_colors = node_df.color.values
    x = node_df.x.values
    y = node_df.y.values
    
    def rescale(val, in_min, in_max, out_min, out_max):
        if in_min == in_max:
            return val
        return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))
    
    x = rescale(x, min(x), max(x), 0.01, .98)
    
    data_trace = dict(
        type='sankey',
        arrangement='snap',
        domain=dict(
            x=[0, 1.], 
            y=[0, 1.]
            ),
        orientation="h",
        valueformat=".0f",
        node=dict(pad=2, thickness=10, line=dict(color="white", width=.5), label=all_nodes, x=x, y=y, color=node_colors),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )
    
    layout = dict(
        height=height,
        width=width,
        margin=go.layout.Margin(l=0, r=0, b=0.1, t=8), #left, right, bottom and top margins
        font=dict(
            size=fontsize, family='Arial',
        )
    )
    
    return data_trace, layout


def run():
    # Get Reactome pathway ids and names
    reactome_pathway_df = get_reactome_pathway_names()
    id_to_name_dict = dict(zip(reactome_pathway_df.reactome_id, reactome_pathway_df.pathway_name))
    name_to_id_dict = dict(zip(reactome_pathway_df.pathway_name, reactome_pathway_df.reactome_id))
    
    nlargest = [10, 10, 10, 10, 6, 6, 6]
    col_name = 'coef'
    
    node_importance = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
    print('Node importance', node_importance[:50])
    #print('\n')
    #print(node_importance.index)
    node_ids = []
    for node in node_importance.index:
        if node in name_to_id_dict.keys():
            node_ids.append(name_to_id_dict[node])
        else:
            node_ids.append(node)
            
    node_importance['node_id'] = node_ids
    #print('\n', node_ids)
    first_layer_nodes = node_importance[node_importance.layer == 1].copy()
    other_layer_nodes = node_importance[node_importance.layer != 1].copy()
    first_layer_high_nodes = get_high_nodes(first_layer_nodes, nlargest=nlargest, column='coef_combined')
    print('First')
    print(first_layer_high_nodes)
    pathways_high_nodes = get_high_nodes(other_layer_nodes, nlargest=nlargest, column='coef')
    print('\n\n Second')
    print(pathways_high_nodes)
    high_nodes = first_layer_high_nodes + pathways_high_nodes
    #print('High nodes', high_nodes)
    high_nodes_df = filter_nodes(node_importance, high_nodes)
    print('Filtered high nodes', high_nodes_df)
    high_nodes_ids = list(high_nodes_df.node_id.values)
    #print('High nodes ids', high_nodes_ids)
    
    links_df = get_links()
    
    """
    MDM4
    """
    mdm4_nodes = get_MDM4_nodes(links_df)
    mdm4_nodes_names = []
    for node in mdm4_nodes:
        if node in id_to_name_dict.keys():
            mdm4_nodes_names.append(id_to_name_dict[node])
        else:
            mdm4_nodes_names.append(node)
            
    idxs = links_df.source == links_df.target
    links_df = links_df[~idxs]
    
    # Keep important nodes only
    links_df = filter_connections(links_df, high_nodes_ids, add_unk=True)
    links_df = links_df.reset_index()
    
    links_df['value_abs'] = links_df.value.abs()

    links_df['child_sum_target'] = links_df.groupby('target').value_abs.transform(np.sum)
    links_df['child_sum_source'] = links_df.groupby('source').value_abs.transform(np.sum)
    links_df['value_normalized_by_target'] = 100 * links_df.value_abs / links_df.child_sum_target
    links_df['value_normalized_by_source'] = 100 * links_df.value_abs / links_df.child_sum_source
    
    node_importance['coef_combined_normalized_by_layer'] = 100. * node_importance[col_name] / node_importance.groupby('layer')[col_name].transform(np.sum)
    node_importance_ = node_importance[['node_id', 'coef_combined_normalized_by_layer', col_name]].copy()
    node_importance_['coef_combined_normalized_by_layer'] = np.log(1. + node_importance_.coef_combined_normalized_by_layer)
    node_importance_normalized = node_importance_[['node_id', 'coef_combined_normalized_by_layer']]
    node_importance_normalized = node_importance_normalized.set_index('node_id')
    node_importance_normalized.columns = ['target_importance']
    
    links_df_ = pd.merge(links_df, node_importance_normalized, left_on='target', right_index=True, how='left')
    node_importance_normalized.columns = ['source_importance']
    links_df_ = pd.merge(links_df_, node_importance_normalized, left_on='source', right_index=True, how='left')
    
    def adjust_values(links_df_in):
        df = links_df_in.copy()
        df['A'] = df.value_normalized_by_source * df.source_importance
        df['B'] = df.value_normalized_by_target * df.target_importance
        df['value_final'] = df[["A", "B"]].min(axis=1)
        #
        df['value_old'] = df.value
        df.value = df.value_final
        #
        df['source_fan_out'] = df.groupby('source').value_final.transform(np.sum)
        df['source_fan_out_error'] = np.abs(df.source_fan_out - 100. * df.source_importance)

        df['target_fan_in'] = df.groupby('target').value_final.transform(np.sum)
        df['target_fan_in_error'] = np.abs(df.target_fan_in - 100. * df.target_importance)
        #
        #
        ind = df.source.str.contains('others')
        df['value_final_corrected'] = df.value_final
        df.loc[ind, 'value_final_corrected'] = df[ind].value_final + df[ind].target_fan_in_error
        ind = df.target.str.contains('others')

        df.loc[ind, 'value_final_corrected'] = df[ind].value_final_corrected + df[ind].source_fan_out_error

        df.value = df.value_final_corrected
        
        return df
    
    df = adjust_values(links_df_)
    important_node_connections_df = df.replace(id_to_name_dict)
    
    high_nodes_df = high_nodes_df[[col_name, 'layer']]
    high_nodes_df.loc['Mutation'] = [1, 0]
    high_nodes_df.loc['Amplification'] = [1, 0]
    high_nodes_df.loc['Deletion'] = [1, 0]
    
    # Add first layer
    first_layer_df = get_first_layer_df(nlargest)
    links_df = pd.concat([first_layer_df, important_node_connections_df], sort=True).reset_index()
    
    links_encoded_df, nodes_df = get_formated_network(links_df, high_nodes_df, col_name=col_name, remove_others=False)
    
    scale = 1.
    width = 600. / scale
    height = 0.5 * width / scale
    data_trace, layout = get_data_trace(links_encoded_df, nodes_df, height, width)
    fig = dict(data=[data_trace], layout=layout)
    fig = go.Figure(fig)
    
    filename = os.path.join(saving_dir, 'sankey_diagram.pdf')
    fig.write_image(filename, scale=1, width=width, height=height, format='pdf')

    filename = os.path.join(saving_dir, 'sankey_diagram.png')
    fig.write_image(filename, scale=5, width=width, height=height, format='png')
    
    scale = 0.5
    width = 600. / scale
    height = 0.5 * width
    data_trace, layout = get_data_trace(links_encoded_df, nodes_df, height, width, fontsize=12)
    fig = dict(data=[data_trace], layout=layout)
    filename = 'sankey_diagram.html'
    filename = os.path.join(saving_dir, filename)
    plot(fig, filename=filename)
    
    
    
if __name__ == "__main__":
    run()