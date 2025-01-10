import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import itertools  # For flattening lists
import torch
import numpy as np

# Define parameters
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Figure sizes
inchmm = 25.4
nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

# Plot options
font = {'family': 'sans',
        'weight': 'normal',
        'size': 10}
plt.rc('font', **font)


def name_edges(weights, layers):
    num_edges = weights.get_edges()
    from_layer = weights.from_layer
    to_layer = weights.to_layer
    named_edges = {}
    for key in num_edges:
        edge_list = []
        for edge in num_edges[key]:
            down, up, weight = edge
            down_name = layers[from_layer].get_node_name(down, from_layer)
            up_name = layers[to_layer].get_node_name(up, to_layer)
            edge_list.append((down_name, up_name, weight))
        named_edges[key] = edge_list
    return named_edges


def name_nodes(layers):
    nodes = {}
    for key in layers:
        nodes[key] = layers[key].get_names(key)
    return nodes


def retrieve_G(layers, weights):
    edges = {}
    for key in weights:
        edges[key] = name_edges(weights[key], layers)
    nodes = name_nodes(layers)
    G = nx.DiGraph()
    for key in nodes:
        G.add_nodes_from(nodes[key], subset=key)
    for edge_set in edges.values():
        for key in edge_set:
            G.add_weighted_edges_from(edge_set[key], color=key)
    return G


def visualize_network(layers, weights, exclude_nodes={},
                      exclude_layers=[], node_size=600,
                      layout='multipartite', show_edge_labels=True,
                      shell_order=None, savefig=False, font_scaling=6,
                      arrow_size=20, device='cpu', **kwargs):
    edges = {}
    for key in weights:
        edges[key] = name_edges(weights[key], layers)

    nodes = name_nodes(layers)
    # Remove specific nodes in specific layers
    for key in exclude_nodes:
        for node in exclude_nodes[key]:
            if node in nodes[key]:
                nodes[key].remove(node)
    # Remove layers using their key
    for key in exclude_layers:
        if key in nodes:
            del nodes[key]
        # Remove corresponding edges
        keys_to_remove = [edge_key for edge_key in edges if key in edge_key]
        for edge_key in keys_to_remove:
            del edges[edge_key]

    G = nx.DiGraph()
    for key in nodes:
        G.add_nodes_from(nodes[key], subset=key)
    for edge_set in edges.values():
        for key in edge_set:
            G.add_weighted_edges_from(edge_set[key], color=key)

    val_map = {'I': 0.5,
               'H': 0.6,
               'O': 0.7}

    values = [val_map.get(node[0], 0.45) for node in G.nodes()]
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    edge_colors = [f"tab:{d['color']}" for _, _, d in G.edges(data=True)]
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]

    if layout == 'multipartite':
        pos = nx.multipartite_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, iterations=10000, threshold=1e-5, **kwargs)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spiral':
        pos = nx.spiral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, **kwargs)
    elif layout == 'shell':
        nlist = []
        if shell_order is None:
            P = len(nodes.keys())
            for key in sorted(nodes.keys()):
                if key < P - 1:
                    nlist.append(nodes[key])
                else:
                    nlist[0] += nodes[key]
            nlist = nlist[::-1]
        else:
            for entry in shell_order:
                if isinstance(entry, list):
                    combined = []
                    for subkey in entry:
                        combined += nodes[subkey]
                    nlist.append(combined)
                else:
                    nlist.append(nodes[entry])
        pos = nx.shell_layout(G, nlist=nlist)
    else:
        print('Sorry, layout not implemented, reverting back to multipartite')
        pos = nx.multipartite_layout(G)

    c = node_size / 600
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Blues'),
                           node_color=values, vmin=0., vmax=1.0,
                           node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=(6 + c * font_scaling))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           arrows=True, arrowsize=arrow_size,
                           width=edge_weights,
                           connectionstyle='arc3,rad=.2')

    if show_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if savefig:
        plt.savefig('network_layout.png', dpi=300)
        plt.savefig('network_layout.svg')

    plt.show()
    return pos

# Additional visualization functions can be similarly ported.
def visualize_dynamic_result(res, columns, time_interval=None):
    if time_interval is not None:
        mask = (res[:, 0] >= time_interval[0]) & (res[:, 0] <= time_interval[1])
        select_res = res[mask]
    else:
        select_res = res

    plt.figure()
    for col in columns:
        plt.plot(select_res[:, 0].cpu().numpy(), select_res[:, col].cpu().numpy(), label=f'Column {col}')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage/Current (V/nA)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_nodes(res, nodes, plot_all=False, onecolumn=False, doublewidth=True,
               time_interval=None):
    """
    Plots the responses of specified nodes over time.

    Parameters
    ----------
    res : torch.Tensor
        Simulation results tensor of shape (num_timesteps, num_columns).
        Assumes the first column is 'Time' and subsequent columns correspond to node data.
    nodes : list
        List of node names corresponding to the columns in `res` (excluding the first 'Time' column).
    plot_all : bool, optional
        If True, plots all available data for each node. Default is False.
    onecolumn : bool, optional
        If True, plots each node in a separate column. Default is False.
    doublewidth : bool, optional
        If True, uses a double-width figure size. Default is True.
    time_interval : tuple, optional
        Tuple specifying the start and end times (e.g., (start_time, end_time)) for plotting.
        If None, plots all available data.
    """

    # Define figure width parameters (ensure these are defined in your notebook)
    # Example:
    # nature_single = 89.0 / 25.4  # Converts mm to inches
    # nature_double = 183.0 / 25.4  # Converts mm to inches

    N = len(nodes)
    Nrows = max(N // 3 + int(bool(N % 3)), 1)  # Maximum of 3 plots per row
    Ncols = min(3, N)

    if onecolumn:
        Nrows = N
        Ncols = 1

    if doublewidth:
        nature_width = nature_double  # Ensure 'nature_double' is defined
    else:
        nature_width = nature_single  # Ensure 'nature_single' is defined

    # Select the appropriate time interval
    if time_interval is not None:
        # Assuming 'res' is a torch tensor with 'Time' in column 0
        mask = (res[:, 0] >= time_interval[0]) & (res[:, 0] <= time_interval[1])
        select_res = res[mask]
    else:
        select_res = res

    # Initialize subplots
    fig, axs = plt.subplots(Nrows, Ncols,
                            figsize=(nature_width * Ncols, nature_single * Nrows))

    # Flatten the axs object using itertools.chain if it's a nested list
    if isinstance(axs, torch.Tensor):
        raise TypeError("'axs' should not be a torch.Tensor")
    elif isinstance(axs, (list, tuple)):
        if isinstance(axs[0], (list, tuple)):
            # axs is a nested list (e.g., list of lists for multiple rows)
            axs_flat = list(itertools.chain.from_iterable(axs))
        else:
            # axs is a flat list
            axs_flat = list(axs)
    else:
        # axs is a single Axes object
        axs_flat = [axs]

    # Plot each node's data
    if N > 1:
        for k, ax in enumerate(axs_flat):
            if k < N:
                node_name = nodes[k]
                # Extract time and node data
                time = select_res[:, 0].cpu().detach().numpy()
                node_data = select_res[:, k + 1].cpu().detach().numpy()  # +1 to skip 'Time' column

                if plot_all:
                    ax.plot(time, node_data, label=f'{node_name} All Data')
                else:
                    ax.plot(time, node_data, label=f'{node_name}')

                ax.set_title(f'Node: {node_name}')
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel('Voltage/Current (V/nA)')
                ax.grid(True)
                ax.legend()
            else:
                # Hide any unused subplots
                ax.axis('off')
    else:
        # Single node plotting
        ax = axs_flat[0]
        node_name = nodes[0]
        time = select_res[:, 0].cpu().detach().numpy()
        node_data = select_res[:, 1].cpu().detach().numpy()  # +1 to skip 'Time' column

        if plot_all:
            ax.plot(time, node_data, label=f'{node_name} All Data')
        else:
            ax.plot(time, node_data, label=f'{node_name}')

        ax.set_title(f'Node: {node_name}')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Voltage/Current (V/nA)')
        ax.grid(True)
        ax.legend()

    # Adjust layout
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    plt.show()