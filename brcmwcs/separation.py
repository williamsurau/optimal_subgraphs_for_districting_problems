import gurobipy
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import brcmwcs.utilities as util


EPSILON = 1e-6


########################################################################################################################
# Callbacks for arc separator

def cb_arc_separator_cuts(model, where):

    if where == gurobipy.GRB.Callback.MIPNODE:
        status = model.cbGet(gurobipy.GRB.Callback.MIPNODE_STATUS)
        if status != gurobipy.GRB.OPTIMAL:
            return
        z = model.cbGetNodeRel(model._myvars['z'])
        y = model.cbGetNodeRel(model._myvars['y'])

    elif where == gurobipy.GRB.Callback.MIPSOL:
        z = model.cbGetSolution(model._myvars['z'])
        y = model.cbGetSolution(model._myvars['y'])
    else:
        return

    z_vars = model._myvars['z']
    y_vars = model._myvars['y']

    backcuts = model._backcuts
    num_nested_cuts = model._num_nested_cuts  # TODO Test 1 or 2
    # gsec = True

    # #################
    # ## draw LP Relax
    # import brcmwcs.draw as draw
    # draw_g = nx.to_undirected(model._full_graph)
    # non_zero_nodes = [v for v, val in y.items() if val > EPSILON and v in draw_g]
    # node_colors = {v: val for v, val in y.items() if val > EPSILON and v in draw_g}
    # draw.draw(draw_g, root=model._root, node_list=non_zero_nodes, node_color_dict=node_colors, colormap='OrRd')
    # #################

    digraph = model._digraph
    g_temp = nx.Graph(digraph)
    if model._is_c2f:
        rem_arcs = [a for a, val in z.items() if val <= 1 - EPSILON and digraph.edges[a]['interior_fine_path']]
        g_temp.remove_edges_from(rem_arcs)
    rem_nodes = [v for v, val in y.items() if val <= 1 - EPSILON]
    g_temp.remove_nodes_from(rem_nodes)
    comps = list(nx.connected_components(g_temp))

    if len(comps) == 1:
        # already connected
        return

    # it is sufficient to check only one node for each connected component
    nodes_to_check = [c.pop() for c in comps if model._root not in c]
    
    # nodes_to_check = [v for v in model._digraph.nodes() if y[v] == 1 and v != model._root]

    ##################

    # def add_gsec(min_cut_arcs):
    #     """
    #     generalized subtour elimination constraints, see Ljubic 2006 PCST
    #     """
    #     r_comp_nodes, _ = util.restricted_bfs(digraph, model._root, set(v for u, v in min_cut_arcs))
    #     non_reachable = set(digraph.nodes).difference(r_comp_nodes)
    #     sum_non_r_arcs = sum(z_vars[u, v] for u in non_reachable for v in digraph[u] if v in non_reachable)
    #     if sum_non_r_arcs:
    #         for u in non_reachable:
    #             model.cbLazy(
    #                 sum_non_r_arcs <= sum(y_vars[w] for w in non_reachable if w != u)
    #             )

    ##################

    node_map = model._node_map
    mapped_root = node_map[model._root]
    mapped_arcs = model._mapped_arcs

    factor = 1e6
    non_zero_entries = [(node_map[u], node_map[v], round(val*factor)+1) if val else (node_map[u], node_map[v], 1)
                        for (u, v), val in z.items()]
    row, col, data = [np.array(l) for l in zip(*non_zero_entries)]
    caps = {(u, v): val for u, v, val in non_zero_entries}

    cut_matrix = csr_matrix((data, (row, col)), shape=(model._num_nodes, model._num_nodes))

    digraph2 = model._digraph2  # digraph2 with node labels converted to integers

    for v in nodes_to_check:

        result = maximum_flow(cut_matrix, mapped_root, node_map[v], method='dinic')
        i = 0

        while result.flow_value/factor - y[v] < -1e-6:
            i += 1

            # retrieve cut
            flow_arr = result.flow.toarray()  # convert to array for faster lookup

            mapped_saturated_arcs = [a for a in mapped_arcs if flow_arr[a[0]][a[1]] == caps[a]]

            mapped_cut_arcs = util.retrieve_cut(digraph2, mapped_root, mapped_saturated_arcs)
            min_cut_arcs = [(model._nodes[i], model._nodes[j]) for i, j in mapped_cut_arcs]

            model._cb_cut_cnt += 1
            model.cbLazy(
                sum(z_vars[a] for a in min_cut_arcs) >= y_vars[v]
            )

            # if gsec:
            #     add_gsec(min_cut_arcs)

            if backcuts:
                mapped_saturated_arcs2 = [a[::-1] for a in mapped_saturated_arcs]
                bla = util.retrieve_cut(digraph2, node_map[v], mapped_saturated_arcs2)
                mapped_cut_arcs2 = [a[::-1] for a in bla]
                min_cut_arcs2 = [(model._nodes[i], model._nodes[j]) for i, j in mapped_cut_arcs2]

                model._cb_cut_cnt += 1
                model.cbLazy(
                    sum(z_vars[a] for a in min_cut_arcs2) >= y_vars[v]
                )

                # if gsec:
                #     add_gsec(min_cut_arcs2)

            if i == num_nested_cuts:
                break

            cut_l_matrix = cut_matrix.tolil()
            for a in mapped_cut_arcs:
                cut_l_matrix[a] = factor
                caps[a] = factor

            cut_matrix = cut_l_matrix.tocsr()

            result = maximum_flow(cut_matrix, mapped_root, node_map[v], method='dinic')


########################################################################################################################
# Callbacks for node separator

def cb_node_separator_integral(model, where):
    """
    Callback that implements the separation routine from
    Fischetti et al. 2017
    "Thinning out Steiner trees: a node-based model for uniform edge costs"

    TODO: ich löse das problem für jede Wurzel neu. interessant wäre aber auch,
    die cuts auf andere Wurzeln zu übertragen.
    """
    if where != gurobipy.GRB.Callback.MIPSOL:
        return

    ###################################################
    def add_cuts(comp_v, separator):
        """
        For each node v of a connected component C with root \notin C
        and for a (minimal) root-C-separator
        we add the cut
                    y(N) >= y_v
        :param comp_v: set of nodes in C
        :param separator: set of nodes in the separator
        """
        sum_sepa = sum([vars[u] for u in separator])
        for v in comp_v:
            model._cb_cut_cnt += 1
            model.cbLazy(
                sum_sepa >= vars[v]
            )

    ###################################################

    graph = model._graph
    root = model._root
    # logger.info(root)
    vars = model._myvars['y']
    backcuts = model._backcuts  # TODO

    y = model.cbGetSolution(vars)
    # selected_nodes = set(v for v, val in y.items() if val > 0.5)
    selected_nodes = set(v for v in graph.nodes if y[v] > 0.5)

    # for debugging:
    # import ksOpt.ksPlot as ksplot
    # ksplot.draw(graph, nodelist=selected_nodes)
    # a = [5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 40, 93, 94, 95, 96, 98, 99, 121, 123, 125, 133, 141, 143, 149, 164, 165, 235]

    # Find the component of the root in the graph induced by selected nodes
    not_selected_nodes = [v for v in graph.nodes if v not in selected_nodes]
    comp_r, comp_r_neighbors = util.restricted_bfs(graph, root, not_selected_nodes)

    if len(comp_r) == len(selected_nodes):
        # the subgraph is connected
        assert set(selected_nodes) == set(comp_r)
        return

    selected_nodes.difference_update(comp_r)
    while selected_nodes:
        v = selected_nodes.pop()

        comp_v, comp_v_neighbors = util.restricted_bfs(graph, v, not_selected_nodes)
        g_comp_r, _ = util.restricted_bfs(graph, root, comp_v)
        minimal_separator = g_comp_r.intersection(comp_v_neighbors)
        add_cuts(comp_v, minimal_separator)
        selected_nodes.difference_update(comp_v)

        if backcuts:
            g_comp_v, _ = util.restricted_bfs(graph, v, comp_r)
            minimal_separator_back = g_comp_v.intersection(comp_r_neighbors)
            if minimal_separator != minimal_separator_back:
                add_cuts(comp_v, minimal_separator_back)

    # print(f'Total cuts in model: {model._cb_cut_cnt}')