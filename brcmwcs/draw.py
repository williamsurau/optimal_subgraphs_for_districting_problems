import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx


def draw_bwr(graph, root=None):
    length_scale = 120

    positions = dict(graph.nodes(data='pos'))
    colors = dict(graph.nodes(data='profit'))

    min_col = min(colors.values())
    max_col = max(colors.values())
    vmin = min(-max_col, min_col)
    vmax = max(max_col, -min_col)

    node_colors = dict(graph.nodes(data='color'))

    # draw red nodes as circles
    nodelist = [n for n, c in node_colors.items() if c == -1]
    red_nodes = draw_bwr_nodes(graph, nodelist, node_shape='o', length_scale=length_scale, vmin=vmin, vmax=vmax)

    # draw blue nodes as triangles
    nodelist = [n for n, c in node_colors.items() if c == 1]
    blue_nodes = draw_bwr_nodes(graph, nodelist, node_shape='^', length_scale=length_scale, vmin=vmin, vmax=vmax)

    nx.draw_networkx_edges(graph, pos=positions)

    # draw root
    if root is None:
        root_l = []
    else:
        root_l = [root]
    root_d = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=root_l,
        node_size=50,
        node_color='yellow',
        linewidths=0.5,
        edgecolors='k',
        node_shape='s'
    )

    # plt.legend(['ok', 1, 2])
    plt.legend([root_d, red_nodes, blue_nodes], [r'root $r$', r'$V_r$', r'$V_b$'],
               loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # sm = cm.ScalarMappable(cmap='bwr', norm=norm)
    sm = cm.ScalarMappable(cmap='PiYG', norm=norm)
    sm._A = []
    plt.colorbar(sm)
    plt.show()


def draw_bwr_nodes(graph, nodelist, node_shape, length_scale, vmin, vmax):
    node_size = []
    node_color = []
    for node in nodelist:
        node_size.append(graph.nodes[node]['length'])
        node_color.append(graph.nodes[node]['profit'])
    mx_node_size = max(node_size)
    node_size = [(size / mx_node_size * length_scale) for size in node_size]
    positions = dict(graph.nodes(data='pos'))

    output = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=nodelist,
        node_size=node_size,
        node_color=node_color,
        cmap='PiYG',
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        edgecolors='k',
        node_shape=node_shape
    )
    return output


def draw_node_weighted_graph2(graph,
                              node_list=None,
                              highlight_node_list=[],
                              root=None,
                              length_scale=120,
                              color_map='rainbow',
                              traffic_map=None,
                              solution_nodes=None):
    positions = {}
    node_size = []
    node_color = []

    if node_list is None:
        node_list = list(graph.nodes())
    if traffic_map is None:
        traffic_map = {v: graph.nodes[v]['traffic'] for v in graph.nodes()}

    min_traf = min(traffic_map.values())
    max_traf = max(traffic_map.values())
    if color_map == 'bwr':
        edgecolors = 'k'
        vmin = min(-max_traf, min_traf)
        vmax = max(max_traf, -min_traf)
    else:
        edgecolors = None
        vmin = 0
        vmax = max_traf

    grey = vmin - 1

    for node in list(graph.nodes()):
        positions[node] = graph.nodes[node]['pos']
        node_size.append(graph.nodes[node]['length'])
        if node in node_list:
            node_color.append(traffic_map[node])
        else:
            node_color.append(grey)
    mx_node_size = max(node_size)
    node_size = [(size / mx_node_size * length_scale) for size in node_size]

    nx.draw_networkx_edges(graph, pos=positions)

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=list(graph.nodes()),
        node_size=node_size,
        node_color=node_color,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        edgecolors=edgecolors
    )

    if highlight_node_list:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=highlight_node_list,
            node_size=[node_size[list(graph.nodes()).index(v)] for v in highlight_node_list],
            node_color='yellow',
            linewidths=0.5,
            edgecolors='k',
        )
    if solution_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=solution_nodes,
            node_size=[node_size[list(graph.nodes()).index(v)] for v in solution_nodes],
            node_color='b',
        )

    if root is not None and root in graph:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=[root],
            node_size=50,
            node_color='yellow',
            linewidths=0.5,
            edgecolors='k',
            node_shape='s'
        )

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm._A = []
    # plt.colorbar(sm)
    plt.box(False)
    plt.show()


def draw(graph,
         root=None,
         node_color_dict=None,
         edge_color_dict=None,
         colormap='PiYG',
         node_shapes=None,
         node_list=None,
         highlight_node_list=None,
         special_node_list=None,
         highlight_edge_list=None,
         ):
    if node_list is None:
        node_list = list(graph.nodes())

    positions = dict(graph.nodes(data='pos'))

    nx.draw_networkx_edges(graph, pos=positions)

    if highlight_edge_list is not None:
        nx.draw_networkx_edges(graph, pos=positions, edgelist=highlight_edge_list, edge_color='red', width=2.5)

    if edge_color_dict is not None:
        edge_list = [e for e in edge_color_dict]
        max_color = max(edge_color_dict.values())
        edge_colors = [edge_color_dict[e]/max_color for e in edge_list]  # from interval [0,1]
        sm = cm.ScalarMappable(cmap=colormap)
        nx.draw_networkx_edges(
            graph,
            pos=positions,
            edgelist=edge_list,
            edge_color=edge_colors,
            edge_cmap=plt.get_cmap(colormap),
            width=5,
        )


    node_sizes = {v: graph.nodes[v]['length'] for v in graph.nodes}
    max_size = max(node_sizes.values())
    node_sizes = {v: s / max_size * 200 for v, s in node_sizes.items()}

    # grey nodes
    grey_nodes = [v for v in graph.nodes if v not in node_list]
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=grey_nodes,
        node_size=[node_sizes[v] for v in grey_nodes],
        node_color='grey',
        linewidths=0.5,
        edgecolors='k'
    )

    # colored nodes
    if node_color_dict is None:
        node_color_dict = {v: graph.nodes[v]['profit'] for v in graph.nodes()}

    min_col = min(node_color_dict.values())
    max_col = max(node_color_dict.values())
    if min_col >= 0 - 1e-6:
        vmin = 0
        vmax = max(max_col, 1)
    else:
        vmin = min(-max_col, min_col)
        vmax = max(max_col, -min_col)

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=node_list,
        node_size=[node_sizes[v] for v in node_list],
        node_color=[node_color_dict[v] for v in node_list],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        edgecolors='k'
    )

    if root is not None and root in graph:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=[root],
            node_size=50,
            node_color='yellow',
            linewidths=0.5,
            edgecolors='k',
            node_shape='s'
        )

    if special_node_list is not None:
        n_list, n_colors = zip(*special_node_list)
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=list(n_list),
            node_size=[node_sizes[v] for v in list(n_list)],
            node_color=list(n_colors),
            linewidths=0.5,
            edgecolors='k',
        )

    if highlight_node_list:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=highlight_node_list,
            node_size=[node_sizes[v] for v in highlight_node_list],
            node_color='yellow',
            linewidths=0.5,
            edgecolors='k',
        )

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm._A = []
    plt.colorbar(sm)
    plt.show()

def draw_network_and_node_labels(graph):
    fig, (instance_ax, label_ax) = plt.subplots(1, 2)
    length_scale = 120
    color_map = 'rainbow'
    positions = {}
    node_size = []
    node_color = []

    node_list = list(graph.nodes())
    traffic_map = {v: graph.nodes[v]['traffic'] for v in graph.nodes()}

    min_traf = min(traffic_map.values())
    max_traf = max(traffic_map.values())
    if color_map == 'bwr':
        vmin = min(-max_traf, min_traf)
        vmax = max(max_traf, -min_traf)
    else:
        vmin = min(0, min_traf)
        vmax = max_traf

    grey = vmin - 1

    for node in list(graph.nodes()):
        positions[node] = graph.nodes[node]['pos']
        node_size.append(graph.nodes[node]['length'])
        if node in node_list:
            node_color.append(traffic_map[node])
        else:
            node_color.append(grey)
    mx_node_size = max(node_size)
    node_size = [(size / mx_node_size * length_scale) for size in node_size]

    nx.draw_networkx_edges(graph, pos=positions, ax=instance_ax)

    for i in range(len(node_list)):
        node_shape = 'o'
        edgecolors = None
        linewidths = 0
        nx.draw_networkx_nodes(G=graph, pos=positions, nodelist=[node_list[i]], node_size=[node_size[i]],
                               node_color=[node_color[i]], node_shape=node_shape, cmap=color_map, vmin=vmin, vmax=vmax,
                               linewidths=linewidths, edgecolors=edgecolors, ax=instance_ax
                               )

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=instance_ax)
    cbar.set_label('Traffic', rotation=270, labelpad=20, size=15)

    instance_ax.axis('off')
    instance_ax.autoscale()

    nx.draw(graph, pos=positions, ax=label_ax, with_labels=True)

    label_ax.axis('off')
    label_ax.autoscale()
    plt.show()


def draw_old_and_new_conflict_pairs(graph, root, old_conflict_pairs, new_conflict_pairs, length_scale=120,
                                    color_map='rainbow', titles=None):
    fig, axs = plt.subplots(2)
    if titles:
        axs[0].set_title(titles[0])
        axs[1].set_title(titles[1])
    positions = {}
    node_size = []
    node_color = []
    node_list = list(graph.nodes())
    profit_map = {v: graph.nodes[v]['profit'] for v in graph.nodes()}
    min_traf = min(profit_map.values())
    max_traf = max(profit_map.values())

    edgecolors = 'k'
    vmin = min(-max_traf, min_traf)
    vmax = max(max_traf, -min_traf)

    grey = vmin - 1

    for node in list(graph.nodes()):
        positions[node] = graph.nodes[node]['pos']
        node_size.append(graph.nodes[node]['length'])
        if node in node_list:
            node_color.append(profit_map[node])
        else:
            node_color.append(grey)
    mx_node_size = max(node_size)
    node_size = [(size / mx_node_size * length_scale) for size in node_size]

    old_conflict_graph = graph.copy()
    old_conflict_graph.remove_edges_from(graph.edges())
    for u, v in old_conflict_pairs:
        old_conflict_graph.add_edge(u, v)
    for u, v in graph.edges():
        old_conflict_graph.add_edge(u, v)
    old_conflict_graph = old_conflict_graph.to_undirected()

    nx.draw_networkx_edges(old_conflict_graph, edgelist=list(graph.edges()), edge_color='grey', width=3.0, pos=positions,
                           ax=axs[0])
    nx.draw_networkx_edges(old_conflict_graph, edgelist=old_conflict_pairs, edge_color='k', width=1.0, pos=positions, ax=axs[0])
    new_conflict_graph = graph.copy()
    new_conflict_graph.remove_edges_from(graph.edges())
    for u, v in new_conflict_pairs:
        new_conflict_graph.add_edge(u, v)
    for u, v in graph.edges():
        new_conflict_graph.add_edge(u, v)
    new_conflict_graph = new_conflict_graph.to_undirected()
    nx.draw_networkx_edges(new_conflict_graph, edgelist=list(graph.edges()), edge_color='grey', width=3.0, pos=positions,
                           ax=axs[1])
    nx.draw_networkx_edges(new_conflict_graph, edgelist=new_conflict_pairs, edge_color='k', width=1.0, pos=positions, ax=axs[1])
    # unique_conflict_pairs = [(u, v) for u, v in new_conflict_pairs
    #                          if (u, v) not in old_conflict_pairs and (v, u) not in old_conflict_pairs]
    # nx.draw_networkx_edges(new_conflict_graph, edgelist=unique_conflict_pairs, edge_color='k', pos=positions, ax=axs[1])

    for ax in axs:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=list(graph.nodes()),
            node_size=node_size,
            node_color=node_color,
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            edgecolors=edgecolors,
            ax=ax
        )
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=[root],
            node_size=50,
            node_color='yellow',
            linewidths=0.5,
            edgecolors='k',
            node_shape='s',
            ax=ax
        )
        ax.set_frame_on(False)
        ax.plot()
    # norm = axs[0].Normalize(vmin=vmin, vmax=vmax)
    # norm = axs[1].Normalize(vmin=vmin, vmax=vmax)
    # sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    # sm._A = []
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(950, 30, 1000, 1000)

    plt.show()

def draw_lp_relaxation(graph, root, solution, length_scale=120, color_map='PiYG'):
    fig, ax = plt.subplots(1)
    # fig, axs = plt.subplots(2)
    positions = {}
    node_size = []
    node_color = []
    node_list = list(graph.nodes())
    profit_map = {v: graph.nodes[v]['profit'] for v in graph.nodes()}
    min_traf = min(profit_map.values())
    max_traf = max(profit_map.values())

    nodelist = list(graph.nodes())

    node_bwr_color = dict(graph.nodes(data='color'))

    edgecolors = 'k'
    vmin = min(-max_traf, min_traf)
    vmax = max(max_traf, -min_traf)

    grey = vmin - 1

    for node in nodelist:
        positions[node] = graph.nodes[node]['pos']
        node_size.append(graph.nodes[node]['length'])
        if node in node_list:
            node_color.append(profit_map[node])
        else:
            node_color.append(grey)
    mx_node_size = max(node_size)
    node_size = [(size / mx_node_size * length_scale) for size in node_size]

    nx.draw_networkx_edges(graph, edge_color='k', pos=positions, ax=ax)

    nodelist_blue = [n for n, c in node_bwr_color.items() if c == -1]
    sizes_blue = [size for i, size in enumerate(node_size)
                  if nodelist[i] in nodelist_blue]
    colors_blue = [color for i, color in enumerate(node_color) if nodelist[i] in nodelist_blue]
    nx.draw_networkx_nodes(
        graph, pos=positions, node_shape='^', node_size=sizes_blue, node_color=colors_blue, nodelist=nodelist_blue,
        cmap=color_map, vmin=vmin, vmax=vmax, linewidths=0.5, edgecolors=edgecolors, ax=ax
    )
    nodelist_red = [n for n, c in node_bwr_color.items() if c == 1]
    sizes_red = [size for i, size in enumerate(node_size)
                 if nodelist[i] in nodelist_red]
    colors_red = [color for i, color in enumerate(node_color) if nodelist[i] in nodelist_red]
    nx.draw_networkx_nodes(
        graph, pos=positions, node_shape='o', node_size=sizes_red, node_color=colors_red, nodelist=nodelist_red,
        cmap=color_map, vmin=vmin, vmax=vmax, linewidths=0.5, edgecolors=edgecolors, ax=ax
    )
    nx.draw_networkx_nodes(
        graph, pos=positions, nodelist=[root], node_size=50, node_color='yellow', linewidths=0.5, edgecolors='k',
        node_shape='s', ax=ax
    )
    ax.set_frame_on(False)
    ax.plot()

    pos_nodes = [v for v in solution if solution[v] > 0]
    label_pos_nodes = {v: round(solution[v], 4) for v in pos_nodes}
    position_pos_nodes = {v: positions[v] for v in pos_nodes}
    pos_graph = nx.Graph()
    pos_graph.add_nodes_from(pos_nodes)
    nx.draw_networkx_labels(pos_graph, pos=position_pos_nodes, font_size=10, labels=label_pos_nodes, ax=ax)

    # nx.draw_networkx_edges(graph, edge_color='k', pos=positions, ax=axs[0])
    # nx.draw_networkx_nodes(
    #     graph, pos=positions, nodelist=list(graph.nodes()), node_size=node_size, node_color=node_color, cmap=color_map,
    #     vmin=vmin, vmax=vmax, linewidths=0.5, edgecolors=edgecolors, ax=axs[0]
    # )
    # nx.draw_networkx_nodes(
    #     graph, pos=positions, nodelist=[root], node_size=50, node_color='yellow', linewidths=0.5, edgecolors='k',
    #     node_shape='s', ax=axs[0]
    # )
    # axs[0].set_frame_on(False)
    # axs[0].plot()

    # pos_nodes = [v for v in solution if solution[v] > 0]
    # label_pos_nodes = {v: round(solution[v], 4) for v in pos_nodes}
    # position_pos_nodes = {v: positions[v] for v in pos_nodes}
    # pos_graph = nx.Graph()
    # pos_graph.add_nodes_from(pos_nodes)
    # nx.draw_networkx_edges(graph, edge_color='k', pos=positions, ax=axs[1])
    # nx.draw_networkx_nodes(
    #     graph, pos=positions, nodelist=list(graph.nodes()), linewidths=0.5,
    #     ax=axs[1]
    # )
    # nx.draw_networkx_labels(pos_graph, pos=positions, labels=label_pos_nodes, ax=axs[1])
    # axs[1].set_frame_on(False)
    # axs[1].plot()

    plt.show()

def draw_solution(problem, solution):
    pass
