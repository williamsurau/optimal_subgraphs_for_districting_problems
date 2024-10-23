import networkx as nx

def restricted_bfs(graph, source, restricted_nodes):
    """
    Fast BFS that does not process restricted_nodes.
    *:returns pair of sets: seen, neighbors
        -seen: set of nodes in the same component
        -neighbors: set of nodes in restricted nodes adjacent to the component
    """
    graph_adj = graph.adj
    seen = set()
    neighbors = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                if v in restricted_nodes:
                    neighbors.add(v)
                    continue
                seen.add(v)
                nextlevel.update(graph_adj[v])
    return seen, neighbors


########################################################################################################################
# connectivity auxiliaries

########################################################
# arc sep
def retrieve_cut(digraph, source, saturated_arcs):
    """
    Fast BFS that does not use cut_arcs.
    *:returns
        set of arcs in saturated_arcs that are reached "first" in BFS
    """
    # implicit edge deletion by only considering unsaturated arcs
    adj = {u: [v for v in n_u if (u, v) not in saturated_arcs] for u, n_u in digraph.adj.items()}

    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(adj[v])

    min_cut_arcs = [(u, v) for u, v in saturated_arcs if u in seen and v not in seen]
    return min_cut_arcs

########################################################
# c2f scf
def compute_c2f_big_m(graph, coarse_graph, root, ub):
    """

    """

    weights = dict(graph.nodes(data='length'))

    nx.set_edge_attributes(coarse_graph, 0, 'weight')
    for a, ifp in coarse_graph.proper_coarse_arcs:
        interior_weight = sum(weights[v_i] for v_i in ifp)
        coarse_graph.edges[a]['weight'] = interior_weight

    coarse_weights = {}
    for v in coarse_graph.nodes:
        if isinstance(coarse_graph, nx.MultiDiGraph):
            min_incident_arc_weight = min(coarse_graph.edges[a]['weight'] for a in coarse_graph.in_edges(v, keys=True))
        else:
            min_incident_arc_weight = min(coarse_graph.edges[a]['weight'] for a in coarse_graph.in_edges(v))
        coarse_weights[v] = weights[v] + min_incident_arc_weight

    sorted_weights = sorted([l for v, l in coarse_weights.items() if v != root])
    new_ub = ub - weights[root]

    big_m = 0
    w_sum = 0
    while sorted_weights and w_sum < new_ub:
        w_sum += sorted_weights.pop(0)
        big_m += 1

    # compute arc specific big M
    spl = nx.single_source_shortest_path_length(coarse_graph, root)

    big_m_dict = {arc: big_m - spl[arc[0]] for arc in coarse_graph.edges}

    return big_m_dict

########################################################
# scf
def compute_scf_big_m(flow_graph, root, ub):
    # compute global big M
    sorted_weights = sorted([l for v, l in flow_graph.nodes(data='length') if v != root])
    new_ub = ub - flow_graph.nodes[root]['length']

    big_m = 0
    w_sum = 0
    while sorted_weights and w_sum < new_ub:
        w_sum += sorted_weights.pop(0)
        big_m += 1

    # compute arc specific big M
    spl = nx.single_source_shortest_path_length(flow_graph, root)

    big_m_dict = {arc: big_m - spl[arc[0]] for arc in flow_graph.edges}

    return big_m_dict


###########################################################
    # Coarse To Fine
def build_coarse_graph(graph, coarse_nodes, allow_multiarcs=True):
    """
    assume that graph has no leaves (that are not coarse nodes)

    """
    if allow_multiarcs:
        coarse_graph = nx.MultiDiGraph(graph)
    else:
        coarse_graph = nx.DiGraph(graph)

    nx.set_edge_attributes(coarse_graph, [], 'interior_fine_path')

    # arcs between coarse nodes:
    coarse_arcs_1 = [(u, v) for u, v in graph.edges if u in coarse_nodes and v in coarse_nodes]
    coarse_arcs_1 += [(v, u) for u, v in coarse_arcs_1]

    # fine path arcs (with actual fine path)
    coarse_graph.coarse_loops = []
    visited = set()
    for u in coarse_nodes:

        for next_node in graph[u]:
            if next_node in visited:
                continue
            # interior fine path (not taking the coarse nodes), maybe empty
            fine_path = [next_node]
            prev_node = u
            while next_node not in coarse_nodes:
                x = [x for x in graph[next_node] if x != prev_node][0]
                prev_node, next_node = next_node, x
                fine_path.append(next_node)

            fine_path = fine_path[:-1]
            if not fine_path:
                # arc between coarse nodes
                continue

            # do not add this from the other direction
            visited.add(prev_node)

            if u == next_node:
                # no loops
                coarse_graph.coarse_loops.append([u] + fine_path + [u])
                continue

            if allow_multiarcs or (u, next_node) not in coarse_arcs_1:
                #  we do not add a multiarc or multiarc is allowed
                coarse_graph.add_edge(u, next_node, interior_fine_path=fine_path)
                coarse_graph.add_edge(next_node, u, interior_fine_path=fine_path[::-1])
                coarse_arcs_1 += [(u, next_node), (next_node, u)]
                del_nodes = fine_path
            else:
                # choose middle node in fine path as coarse node to prevent multiarcs
                mid_index = int(len(fine_path) / 2)
                v = fine_path[mid_index]
                first_fine_path = fine_path[:mid_index]
                second_fine_path = fine_path[mid_index + 1:]

                coarse_graph.add_edge(u, v, interior_fine_path=first_fine_path)
                coarse_graph.add_edge(v, u, interior_fine_path=first_fine_path[::-1])
                coarse_graph.add_edge(v, next_node, interior_fine_path=second_fine_path)
                coarse_graph.add_edge(next_node, v, interior_fine_path=second_fine_path[::-1])
                del_nodes = first_fine_path + second_fine_path

            coarse_graph.remove_nodes_from(del_nodes)

    # proper coarse arcs (with non-empty fine path)
    if isinstance(coarse_graph, nx.MultiDiGraph):
        proper_ca = [(a, coarse_graph.edges[a]['interior_fine_path']) for a in coarse_graph.edges(keys=True)
                     if coarse_graph.edges[a]['interior_fine_path']]
    else:
        proper_ca = [(a, coarse_graph.edges[a]['interior_fine_path']) for a in coarse_graph.edges
                     if coarse_graph.edges[a]['interior_fine_path']]
    coarse_graph.proper_coarse_arcs = proper_ca

    return coarse_graph


def get_fine_arcs(coarse_graph, mode):
    """
    :param mode: 'pred' or 'big_m': what is the value of fine_arcs dict? predecessor of arc or big M?
    """
    assert(mode in ['pred', 'big_m'])

    fine_nodes = set()
    fine_arcs = {}

    for a, ifp in coarse_graph.proper_coarse_arcs:
        u = a[0]

        fine_nodes.update(ifp)

        if mode == 'pred':
            ifp = [None, u] + ifp
            for i, x in enumerate(ifp[1:-1]):
                fine_arcs[x, ifp[i + 2]] = ifp[i]
        else:
            ifp_len = len(ifp)
            ifp = [u] + ifp
            for i, x in enumerate(ifp[:-1]):
                fine_arcs[x, ifp[i + 1]] = ifp_len - i

    for loop in coarse_graph.coarse_loops:
        u = loop[0]  # the coarse node
        if mode == 'pred':
            loop = [None] + loop + [None]
            for i, x in enumerate(loop[1:-1]):
                pred, succ = loop[i], loop[i+2]
                if succ is not None and succ != u:
                    fine_arcs[x, succ] = pred
                if pred is not None and pred != u:
                    fine_arcs[x, pred] = succ
        else:
            loop_len = len(loop) - 2
            for i, x in enumerate(loop[:-1]):
                v = loop[i+1]
                if v != u:
                    fine_arcs[x, v] = loop_len - i
                if x != u:
                    fine_arcs[v, x] = i

    return fine_arcs, fine_nodes
