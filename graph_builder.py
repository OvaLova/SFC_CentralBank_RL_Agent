import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


def build_dependency_graph(dependencies):
    G = nx.DiGraph()
    for var, deps in dependencies:
        for dep in deps:
            G.add_edge(dep, var)
    return G

def build_condensation_graph(G):
    sccs = list(nx.strongly_connected_components(G))
    cond_graph = nx.condensation(G, sccs)
    return cond_graph, sccs

def visualize_dependency_graph(G, sccs, title="Dependency Graph", filename="dependency_graph.pdf"):
     # Assign a color to each node based on its SCC
    scc_map = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            scc_map[node] = i
    # Generate distinct colors
    node_colors = []
    red_shades = list(mcolors.LinearSegmentedColormap.from_list("reds", ["lightcoral", "darkred"])(i / max(1, len(sccs) - 1)) for i in range(len(sccs)))
    for node in G.nodes:
        scc_idx = scc_map[node]
        if len(sccs[scc_idx]) == 1:
            node_colors.append("lightgreen")
        else:
            node_colors.append(red_shades[scc_idx])
    # Draw
    plt.figure(figsize=(20, 12))
    # pos = hierarchical_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
            node_size=300, font_size=5, font_weight='bold', arrowsize=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf"):
    # Map SCC indices to labels
    label_map = {i: " | ".join(sorted(scc)) for i, scc in enumerate(sccs)}
    # Assign colors
    red_cmap = mcolors.LinearSegmentedColormap.from_list("reds", ["lightcoral", "darkred"])
    node_colors = []
    red_index = 0
    for i, scc in enumerate(sccs):
        if len(scc) == 1:
            node_colors.append("lightgreen")
        else:
            node_colors.append(red_cmap(red_index / max(1, len(sccs) - 2)))
            red_index += 1  # Only increment for actual cyclic SCCs
    # Draw
    plt.figure(figsize=(20, 12))
    # pos = hierarchical_layout(cond_graph)
    pos = nx.nx_agraph.graphviz_layout(cond_graph, prog="dot")
    nx.draw(cond_graph, pos, with_labels=True, labels=label_map,
            node_color=node_colors, edge_color='gray',
            node_size=300, font_size=4, font_weight='bold', arrowsize=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def hierarchical_layout(G, vertical_gap=1.5, horizontal_gap=1.5):
    # Compute depth (longest path from a source)
    depths = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            depths[node] = max(depths[p] for p in preds) + 1
        else:
            depths[node] = 0
    # Group by depth
    layers = defaultdict(list)
    for node, depth in depths.items():
        layers[depth].append(node)
    # Assign positions: (x, y), layer by layer
    pos = {}
    for depth, nodes in layers.items():
        x_spacing = horizontal_gap * (len(nodes) - 1)
        for i, node in enumerate(nodes):
            x = -x_spacing / 2 + i * horizontal_gap
            y = -depth * vertical_gap
            pos[node] = (x, y)
    return pos

