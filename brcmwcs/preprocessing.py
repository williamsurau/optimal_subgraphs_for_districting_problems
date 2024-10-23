import copy
import itertools
import json
import logging
import time

import networkx as nx
import numpy as np

import brcmwcs.utilities as util
import brcmwcs.draw as draw

logger = logging.getLogger()

EPSILON = 1e-6


class PrepOutput:

    def __init__(self, preprocessing=None):

        self.fixed_nodes = getattr(preprocessing, 'fixed_nodes', set())
        self.removed_nodes = getattr(preprocessing, 'removed_nodes', set())
        self.implied_nodes = getattr(preprocessing, 'implied_nodes', {})
        self.activation_pairs = getattr(preprocessing, 'activation_pairs', set())
        self.forbidden_pairs = getattr(preprocessing, 'forbidden_pairs', set())
        self.root_rings = getattr(preprocessing, 'root_rings', [])
        self.lb_disk_nodes = getattr(preprocessing, 'lb_disk_nodes', None)
        self.feasible = getattr(preprocessing, 'root_is_possible', True)

        self.time = getattr(preprocessing, 'time', 0)
        self.stats = getattr(preprocessing, 'stats', None)

        if preprocessing is not None:
            fw = preprocessing.bicolor_fixed_sum
            wr = preprocessing.weight_ranges
            if wr is None:
                self.color_range = None
            else:
                self.color_range = {-1: [wr[0][0] + fw[0], wr[1][0] + fw[0]],
                                    1: [wr[0][1] + fw[1], wr[1][1] + fw[1]]}
        else:
            self.color_range = None

    def save_to_json(self, filepath):
        for v in self.implied_nodes:
            self.implied_nodes[v] = list(self.implied_nodes[v])
        json_solution = {'fixed_nodes': list(self.fixed_nodes),
                         'removed_nodes': list(self.removed_nodes),
                         'implied_nodes': self.implied_nodes,
                         'activation_pairs': list(self.activation_pairs),
                         'forbidden_pairs': list(self.forbidden_pairs),
                         'feasible': int(self.feasible),
                         'color_range': list(self.color_range),
                         'time': self.time}
        with open(filepath, 'w') as f:
            json.dump(json_solution, f, indent=2)
        logger.info('\nprep_out written to {}\n'.format(filepath))

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            json_prep = json.load(f)

        self.fixed_nodes = json_prep['fixed_nodes']
        self.removed_nodes = json_prep['removed_nodes']
        self.implied_nodes = json_prep['implied_nodes']
        self.implied_nodes = {int(v): x for v, x in self.implied_nodes.items()}
        self.activation_pairs = json_prep.get('activation_pairs', [])
        self.forbidden_pairs = json_prep['forbidden_pairs']
        self.feasible = json_prep['feasible']
        self.color_range = json_prep['color_range']
        self.time = json_prep['time']

        logger.info('\nprep_out loaded from {}\n'.format(json_path))


class PrepStats:

    def __init__(self):

        self.total_time = None

        # time for remove and fix nodes
        self.basic_time = None

        self.implied_nodes_time = None
        self.num_implications_init = None

        self.filter_implied_node_time = None
        self.num_implications_filtered = None  # how many implications after filter

        self.conflict_pair_time = None
        self.num_conflicts_init = None
        self.num_conflicts_filtered = None


class Preprocessing:
    """
    Most of the color related values are organized as pairs (blue, red).

    """

    def __init__(self, problem, prep_out=None):

        self.problem = problem

        #################

        # nodes that have to be in any feasible template
        self.fixed_nodes = set() if prep_out is None else prep_out.fixed_nodes

        # nodes that cannot be in any feasible template
        self.removed_nodes = set() if prep_out is None else prep_out.removed_nodes

        # v -> set of nodes that are used if v is used
        self.implied_nodes = {v: set() for v in problem.graph.nodes} if prep_out is None else prep_out.implied_nodes

        # list of node pairs where at least one node is chosen in any solution
        self.activation_pairs = []

        # set of node pairs that cannot be both part of a feasible solution
        self.forbidden_pairs = set() if prep_out is None else prep_out.forbidden_pairs

        #################
        # we manipulate the problem graph
        self.graph = problem.graph  # nx.Graph instance

        # [lb_range, ub_range] as np.arrays
        self.weight_ranges = None

        # bicolor stuff
        # node -> weight pair (one entry is 0.)
        self.bicolor_weights = {}

        # bicolor weights for fixed nodes are set to (0.,0.), but we need the original for node fixing
        self.bicolor_weights_orig = {}

        # arc (u,v) -> weight pair of node v
        # TODO essentially used for virtual deletion of nodes. Maybe do this differently.
        self.bicolor_arc_weights = {}

        # cumulated (unfixed) weight
        self.bicolor_sum = None

        # cumulated fixed weight
        self.bicolor_fixed_sum = np.array((0., 0.))

        # auxiliary
        self.fixed_weights = {}

        # set during bicolor_radius
        self.bicolor_labels = {}
        self.bicolor_on_path = {}

        self.core_graph = None
        self.articulation_points = []

        # arc rings around the root where one arc of each ring has to be chosen
        self.root_rings = []

        # nodes within lb radius of root
        self.lb_disk_nodes = None

        self.time = 0

        self.stats = PrepStats()

        # can we build a feasible template from root (as median)
        self.root_is_possible = True

        # thresholds for filter conflict
        self.threshold = {
            0: 0,
            1: 0,
            2: 0,
            3: 0.1,
            4: 0.2,
            5: 0.3,
            6: 0.4,
            7: 0.5,
            8: 0.6,
        }

    def setup(self):
        """
        Compute the weight ranges and setup data structures.
        """
        for v, c in self.graph.nodes(data='color'):
            weight = self.graph.nodes[v]['length']
            bicolor_weight = np.array((0., weight)) if c == 1 else np.array((weight, 0.))
            self.bicolor_weights[v] = bicolor_weight
            self.bicolor_weights_orig[v] = bicolor_weight

        self.bicolor_sum = sum(self.bicolor_weights.values())
        self.bicolor_fixed_sum = np.array((0., 0.))

        # compute weight ranges
        self.compute_weight_ranges()

        # setup bicolor data
        self.bicolor_labels = {v: [] for v in self.graph.nodes}
        root = self.problem.root
        self.bicolor_labels[root].append((np.array((0., 0.)), set()))

        self.bicolor_arc_weights = {u: {} for u in self.graph.nodes}
        for v in self.graph.nodes:
            for u in self.graph[v]:
                self.bicolor_arc_weights[u][v] = self.bicolor_weights[v]
        # remove arcs towards root
        for v in self.graph[root]:
            del self.bicolor_arc_weights[v][root]

        self.bicolor_on_path = {}

    def compute_weight_ranges(self):
        delta = self.problem.delta
        lb_length = self.problem.lb_length
        ub_length = self.problem.ub_length

        self.weight_ranges = [
            # lower bounds:
            np.array((
                max((lb_length - delta) / 2,  # from lower bound
                     self.bicolor_fixed_sum[1] - delta,  # from balancing
                     self.bicolor_fixed_sum[0])  # 0 is an obvious lower bound (if very much is fixed)
                - self.bicolor_fixed_sum[0],
                max((lb_length - delta) / 2,  # from lower bound
                    self.bicolor_fixed_sum[0] - delta,  # from balancing
                    self.bicolor_fixed_sum[1])  # 0 is an obvious lower bound (if very much is fixed)
                - self.bicolor_fixed_sum[1]
            )),
            # upper bounds:
            np.array((
                min((ub_length + delta) / 2,  # from upper bound
                    self.bicolor_sum[1] + self.bicolor_fixed_sum[1] + delta,  # from balancing
                    self.bicolor_sum[0] + self.bicolor_fixed_sum[0])  # natural
                - self.bicolor_fixed_sum[0],
                min((ub_length + delta) / 2,  # from upper bound
                     self.bicolor_sum[0] + self.bicolor_fixed_sum[0] + delta,  # from balancing
                     self.bicolor_sum[1] + self.bicolor_fixed_sum[1])  # natural
                - self.bicolor_fixed_sum[1]
            ))
        ]

    def get_bicolor_cutoff(self):
        return self.weight_ranges[1]

    def check_weights(self, color_weights):
        """
        check if given red weights and blue weights fulfill lower bound
        """
        if any(color_weights < self.weight_ranges[0] - EPSILON):
            return False
        return True

    def run_preprocessing(self):
        """

        """
        prep_start = time.time()

        self.setup()

        if not self.check_weights(self.bicolor_sum):
            # initial feasibility check
            self.root_is_possible = False
            return self.stop_preprocessing(prep_start)

        root_possible, remove_nodes = self.discard_nodes()

        if not root_possible:
            return self.stop_preprocessing(prep_start)

        fixed_nodes = True
        while fixed_nodes:

            root_possible, fixed_nodes = self.fix_nodes()

            if root_possible and fixed_nodes:
                # since we started with discarding, we only repeat it if anything was fixed
                root_possible, removed_nodes = self.discard_nodes(fixed_nodes)
                if not removed_nodes:
                    # if nothing was discarded, we stop the loop
                    fixed_nodes = False

            if not root_possible:
                self.root_is_possible = False
                return self.stop_preprocessing(prep_start)

        self.stats.basic_time = round(time.time() - prep_start, 1)
        logger.debug(f'\n{len(self.removed_nodes)} removed nodes:\n{self.removed_nodes}\n'
                     f'{len(self.fixed_nodes)} / {self.graph.number_of_nodes()} fixed nodes:\n{self.fixed_nodes}')

        # save articulation points at graph for later use
        self.graph.articulation_points = self.articulation_points

        self.find_network_induced_cuts()
        self.build_core_graph()

        # has core graph more than one node?
        proper_core_graph = self.core_graph.number_of_nodes() > 1

        if not proper_core_graph:
            return self.stop_preprocessing(prep_start)

        if self.problem.params.find_implied_nodes:
            self.find_implied_nodes()

        # self.find_activation_pairs()

        # analyse and visualize dominating and non dominating conflict pairs
        # plot_dominating_conflicts(self)
        if self.problem.params.find_conflict_pairs_mode:
            if self.problem.params.find_conflict_pairs_mode == 3:
                self.forbidden_pairs = self.find_conflict_pairs()
            else:
                self.forbidden_pairs = self.find_forbidden_pairs_old()

            if self.problem.params.filter_conflicts:
                self.filter_conflict_pairs()

        # draw.draw(self.graph, self.problem.root, highlight_edge_list=self.forbidden_pairs)
        self.compute_root_rings()
        self.compute_lb_disk()

        return self.stop_preprocessing(prep_start)

    def stop_preprocessing(self, prep_start):

        self.time = round(time.time() - prep_start, 1)
        self.stats.total_time = self.time
        logger.debug(f'Preprocessing finished after {self.time} seconds.')

        # print detailed times if available
        for att, val in vars(self.stats).items():
            if val is not None:
                logger.debug(f'\t{att:>30}: {val:>20}')

        prep_out = PrepOutput(self)
        return prep_out

    def check_bicolor_labels(self, labels):
        cutoff = self.get_bicolor_cutoff()
        labels_to_remove = []
        for v, l_list in labels.items():
            if v == self.problem.root or v in self.fixed_nodes:
                continue
            for i, (label, path) in enumerate(l_list):
                if any(label > cutoff + EPSILON):
                    labels_to_remove.append((v, i))

        # consider in reversed order to not mess with index deletion
        for v, i in labels_to_remove[::-1]:
            del labels[v][i]

    ####################################################################################################################
    # discard nodes

    def discard_nodes(self, fixed_nodes=None):
        """
        we discard all nodes outside of the bicolor radius, i.e.,
        nodes that cannot be reached by any feasible path from
        the root node.

        :return: bool: is instance still feasible after removal?
        """
        removed_nodes = False
        labels = self.compute_bicolor_labels(fixed_nodes)
        nodes_to_remove = set(v for v, l in labels.items() if not l)
        while nodes_to_remove:
            removed_nodes = True
            self.remove_nodes(nodes_to_remove)
            self.check_bicolor_labels(labels)
            nodes_to_remove = set(v for v, l in labels.items() if not l)

        is_feasible = self.check_weights(self.bicolor_sum)
        if not is_feasible:
            self.root_is_possible = False
            return False, False

        # v -> set of nodes that are reached via pareto optimal path via v
        on_pareto_path = {v: set() for v in self.bicolor_labels}
        for v, l_list in self.bicolor_labels.items():
            pareto_path_nodes = {v}.union(*[l[1] for l in l_list])
            for u in pareto_path_nodes:
                on_pareto_path[u].add(v)
        self.bicolor_on_path = on_pareto_path

        return is_feasible, removed_nodes

    def remove_nodes(self, nodes_to_remove):
        for v in nodes_to_remove:
            if v in self.removed_nodes:
                continue

            self.bicolor_sum -= self.bicolor_weights[v]


            # color = self.node_colors[v]
            # self.br_nodes[color].remove(v)
            # self.br_weights[color] -= self.weight[v]
            #

            del self.bicolor_labels[v]
            del self.bicolor_arc_weights[v]
            for u in self.graph[v]:
                if u in self.bicolor_arc_weights and v in self.bicolor_arc_weights[u]:
                    del self.bicolor_arc_weights[u][v]

        self.compute_weight_ranges()

        self.graph.remove_nodes_from(nodes_to_remove)
        self.removed_nodes.update(nodes_to_remove)

    def _add_bicolor_label(self, label, v, labels):
        """
        :param label: pair of bicolor weight and path
        :param v: node where to potentially add this label
        :param labels: all current labels
        :return: bool: is label not dominated?
        """
        lab, path = label
        l_list = labels[v]

        if any(all(l <= lab + EPSILON) for l, p in l_list):
            # label is dominated and should not be added
            return False

        # keep only non-dominated labels
        labels[v] = [(l, p) for l, p in l_list if any(l < lab + EPSILON)] + [label]
        return True

    def _bicolor_radius(self, labels, lengths, updated_labels, cutoff=None, stop_labels=None):

        add_label_fct = self._add_bicolor_label

        # upper color bounds
        if cutoff is None:
            cutoff = self.get_bicolor_cutoff()

        sum_fixed_weights = self.bicolor_fixed_sum.sum()
        while updated_labels:
            new_updated_labels = []
            for u, lab, path in updated_labels:
                for v, l in lengths[u].items():
                    new_label = lab + l
                    if any(new_label > cutoff + EPSILON):
                        # the new label contradicts one of the weight ranges
                        continue
                    if sum(new_label) + sum_fixed_weights > self.problem.params.ub_length + EPSILON:
                        continue
                    # accept new label if not dominated
                    new_path = path.union({u})
                    # if self._add_bicolor_label((new_label, new_path), v, labels):
                    if add_label_fct((new_label, new_path), v, labels):
                        new_updated_labels.append((v, new_label, new_path))

            # if stop_labels is not None:
            #     self.debug_plot(set(l[0] for l in updated_labels))

            if stop_labels is not None and all(labels[v] for v in stop_labels):
                return []
            updated_labels = new_updated_labels

    def _heur_bicolor_radius(self, labels, lengths, start_arcs, stop_labels=None):
        """
        this is a quick way to guarantee that all nodes are still reachable when
        a single node is removed. The heuristic is called in find_implied_nodes
        and only if it cannot label all vertices with feasible labels, we invoke
        the exact method (i.e. when this functions returns False).
        """

        def add_label(label, v):
            for i, l in enumerate(labels[v]):
                if all(l <= label):
                    # label is dominated and should not be added
                    return False
                elif all(l >= label):
                    # label dominates label l -> remove l from labels
                    labels[v].pop(i)

            labels[v].append(label)
            return True

        # upper color bounds
        cutoff = self.get_bicolor_cutoff()

        while start_arcs:
            new_start_arcs = []
            # self.debug_plot(set(a[0] for a in start_arcs).union(set(a[1] for a in start_arcs)))
            for u, v, lab in start_arcs:
                l = lengths[u][v]
                new_label = lab + l
                if any(new_label > cutoff):
                    # the new label contradicts one of the weight ranges
                    continue

                # accept new label if not dominated
                if add_label(new_label, v):
                    for w in lengths[v]:
                        if w != u and w in stop_labels:
                            new_start_arcs.append((v, w, new_label))

            if stop_labels is not None and all(labels[v] for v in stop_labels):
                return True
            start_arcs = new_start_arcs
        return False

    def compute_bicolor_labels(self, newly_fixed_nodes=None):
        """
        Here, we use the red and the blue upper weight bound together with a node labelling approach
        similar to a constrained shortest path to find which nodes cannot be reached.
        We also save all pareto optimal paths.
        """
        labels = self.bicolor_labels
        updated_labels = []
        if newly_fixed_nodes is None:
            # this is the initial run

            # list to efficiently find updated labels
            # labels are triple: node, (blue weight, red weight), path (as set)
            updated_labels = [(self.problem.root, np.array((0., 0.)), set())]
        else:
            # if some nodes were fixed during the last run, we see which labels are now
            # affected and recompute all labels from these changed labels to save some
            # time.

            # decrease the labels affected from recent node fixes
            affected_nodes = set().union(*[self.bicolor_on_path[v] for v in newly_fixed_nodes])
            for v in affected_nodes:
                # base diff is necessary because v is not part of the stored path to v
                base_diff = -self.fixed_weights[v] if v in newly_fixed_nodes else np.array((0, 0))
                temp_labels = list(labels[v])  # copy labels to avoid issues
                for label, path in temp_labels:
                    l_diff = base_diff + sum(-self.fixed_weights[w] for w in path.intersection(newly_fixed_nodes))
                    # label += l_diff
                    new_label = label + l_diff
                    if self._add_bicolor_label((new_label, path), v, labels):
                        updated_labels.append((v, new_label, path))

            # adjust labels
            cutoff = self.get_bicolor_cutoff()
            for v, l_list in labels.items():
                valid_labels = [l for l in l_list if all(l[0] <= cutoff + EPSILON)]
                labels[v] = valid_labels

            # even if all vertices still have labels, we should run the
            # algorithm to find if there are new pareto-optimal ones.

        self._bicolor_radius(labels, self.bicolor_arc_weights, updated_labels)
        return labels

    ####################################################################################################################
    # fix nodes
    def fix_single_node(self, v):
        """
        append v to fixed nodes, reduce the bounds on the weight_ranges
        """
        if v in self.fixed_nodes:
            return

        self.fixed_nodes.add(v)

        self.fixed_weights[v] = self.bicolor_weights[v]
        self.bicolor_fixed_sum += self.bicolor_weights[v]
        self.bicolor_sum -= self.bicolor_weights[v]

        self.bicolor_weights[v] = np.array((0., 0.))

        self.compute_weight_ranges()
        for u in self.graph[v]:
            # from now on, we do not count this weight in bicolor radius (but adjust color_range)
            self.bicolor_arc_weights[u][v] = np.array((0, 0))

    def _fix_nodes(self, virtual_removed_nodes=None, visited=None):
        """
        :return:
        """
        # v -> weight of root component in G-v (only if infeasible)
        root_comp_wo = {}
        root = self.problem.root

        if virtual_removed_nodes is None:
            virtual_removed_nodes = []
        if visited is None:
            visited = {root}

        # my DFS (adapted from nx)
        def add_neighbors_to_stack(u, v, stack):
            # add neighbors of v other than u to stack
            for w in self.graph[v]:
                if w != u and w not in virtual_removed_nodes:
                    stack.append((v, w))

        root_comp = []
        stack = [(root, w) for w in self.graph[root] if w not in virtual_removed_nodes]
        while stack:
            u, v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            if v in self.fixed_nodes:
                add_neighbors_to_stack(u, v, stack)
                if root_comp:
                    root_comp_weight += self.bicolor_weights_orig[v]
                continue
            if v not in self.articulation_points:
                # check if this single node is necessary
                cum_weight = self.bicolor_sum - self.bicolor_weights[v]
                if not self.check_weights(cum_weight):
                    if not virtual_removed_nodes:
                        # only actually fix when called from fix_nodes()
                        self.fix_single_node(v)
                    root_comp_wo[v] = cum_weight
                add_neighbors_to_stack(u, v, stack)
                continue

            if not root_comp:
                root_comp, _ = util.restricted_bfs(self.graph, root, [v] + virtual_removed_nodes)
                root_comp_weight = sum(self.bicolor_weights_orig[n] for n in root_comp)

            # subtract bicolor fixed sum because the weight range assumes fixed weights to be 0.
            if self.check_weights(root_comp_weight - self.bicolor_fixed_sum):
                # if the root_comp is feasible, do not go further in DFS
                root_comp = []
                continue

            root_comp_wo[v] = root_comp_weight.copy()
            if not virtual_removed_nodes:
                # only actually fix when called from fix_nodes()
                self.fix_single_node(v)
            # else:
            root_comp_weight += self.bicolor_weights_orig[v]

            add_neighbors_to_stack(u, v, stack)
            if self.graph.degree(v) != 2:
                # reset root_comp
                root_comp = []

        return root_comp_wo

    def fix_nodes(self):
        """
        Here, we determine which nodes have to be included in any solution.
        Therefore, we determine the articulation points of the graph, and
        perform a depth-first search from the root.
        Any considered node v that is an articulation point is removed from
        the graph and we check if the root-component is feasible.
        If not, node v has to be part of any solution and we subtract its
        weight from the lower and upper weight bound on the corresponding
        color.
        The DFS makes the procedure more efficient s.t. we do not have
        to compute the root-component too often.
        """

        self.articulation_points = set(nx.articulation_points(nx.to_undirected(self.graph)))

        root_comp_wo = self._fix_nodes()
        fixed_nodes = set(root_comp_wo)

        upper_weight_bound_ok = all(self.get_bicolor_cutoff() > -EPSILON)

        # g2 = nx.to_undirected(self.graph)
        # import ksOpt.ksPlot as ksplot
        # cols = {0: 'gray', 1: 'lime'}
        # no_co = [cols[n in self.fixed_nodes] for n in self.graph.nodes]
        # ksplot.draw(g2, nodelist=self.graph.nodes(), node_color=no_co)

        return upper_weight_bound_ok, fixed_nodes

    ####################################################################################################################
    # Core Graph

    def find_network_induced_cuts(self):
        """
        If u is an articulation node, and v is a neighbor in a non-root component
        then the corresponding network induced cut (nic) is
            y_v <= y_u
        If v is chosen, then u has to be chosen as well.

        We find these by performing a BFS from the root and storing the
        predecessors. Then we flip this into successors info and determine
        the articulation nodes.
        We exploit that an articulation point u is the unique predecessor of v.
        """
        preds = nx.predecessor(self.graph, self.problem.root)
        succs = {v: [] for v in self.graph}
        for v, pred_list in preds.items():
            for u in pred_list:
                succs[u].append(v)

        if not self.articulation_points:
            self.articulation_points = set(nx.articulation_points(nx.to_undirected(self.graph)))

        for u in self.articulation_points:
            for v in succs[u]:
                self.implied_nodes[v].add(u)

    def build_core_graph(self):
        """
        If network induced cuts were introduced, we do not need flow constraints
        for the outer branches (i.e. leaves + paths to degree >2 nodes).
        The graph without these branches is the core graph
        Expl:
            E = 1,3  2,3  3,4  4,5  4,6  5,6  6,7, root=4
            E(core graph) =   4,5  4,6  5,6

        :return core_graph:    nx.Graph
        """

        logger.debug('Building core graph')
        core_graph = nx.Graph(self.graph)
        root = self.problem.root

        leaves = [v for v in core_graph.nodes if core_graph.degree(v) == 1]

        while leaves:
            v = leaves.pop()
            if v == root:
                continue

            u = next(core_graph.neighbors(v))
            core_graph.remove_node(v)
            if core_graph.degree(u) == 1:
                leaves.append(u)

        self.core_graph = core_graph
        self.problem.core_graph = core_graph

        return core_graph

    ####################################################################################################################
    # conditional fixes

    def find_implied_nodes(self):
        """
        u in T => v in T

        For each coarse node v, we check which nodes are within the bicolor radius in G-v.
        To this end, we look up for which nodes u the node v was on a pareto optimal
        path. We delete all labels belonging to such paths from this node and start
        the bicolor radius with an appropriate set of original labels and start edges.

        :return:
        """
        start_time = time.time()
        num_impl_found = 0

        # coarse_nodes = set(v for v, d in self.graph.degree if d > 2)
        non_coarse_nodes = set(v for v, d in self.graph.degree if d <= 2)

        done = self.fixed_nodes  # und vllt noch mehr TODO
        for _, v in nx.bfs_edges(self.core_graph, self.problem.root):
            if v in done or v in non_coarse_nodes:
                continue

            labels = {}
            new_label_nodes = self.bicolor_on_path[v]
            all_neighbors = set()
            for w, l_list in self.bicolor_labels.items():
                if w not in new_label_nodes:
                    labels[w] = l_list
                    # if v is not on any pareto optimal path to w, we keep all labels
                    continue

                # otherwise we remove all labels with corresponding path over v
                labels[w] = [l for l in l_list if v not in l[1]]

                # and mark neighboring vertices with unchanged labels
                # start_neighbors = set(u for u in self.graph[w] if u not in new_label_nodes)
                all_neighbors.update(self.graph[w])

            all_neighbors.discard(v)
            non_label_nodes = [w for w, l in labels.items() if not l and w in self.core_graph]
            if not non_label_nodes:
                # all nodes can still be reached (except for non-core graph nodes which we ignore here)
                continue

            lengths = copy.deepcopy(self.bicolor_arc_weights)
            for w in self.graph[v]:
                # assure that node v cannot be reached (implicit removal)
                del (lengths[w][v])

            # first we try a heuristic approach
            ###################################
            heur_labels = {w: [l[0] for l in llist] for w, llist in labels.items()}
            start_arcs = []
            for w in non_label_nodes:
                for u in lengths[w]:
                    if w in lengths[u]:
                        start_arcs += [(u, w, l) for l in heur_labels[u]]

            all_labeled = self._heur_bicolor_radius(heur_labels, lengths, start_arcs, stop_labels=non_label_nodes)

            if all_labeled:
                # the simple heuristic was able to proof that still every node is reachable
                continue

            # if not enough, we do the full algorithm
            #########################################
            updated_labels = []
            for w in all_neighbors:
                updated_labels += [(w, *l) for l in labels[w]]

            self._bicolor_radius(labels, lengths, updated_labels, stop_labels=non_label_nodes)
            infeasible_nodes = set(v for v, l in labels.items() if not l)

            for w in infeasible_nodes:
                self.implied_nodes[w].add(v)
                num_impl_found += 1
            # if infeasible_nodes:
            #     logger.debug(f'\t{v} <= any of {infeasible_nodes}')

        total_time = round(time.time() - start_time, 1)
        num_impl = sum(len(v) for v in self.implied_nodes.values())
        logger.debug(f'Found {num_impl} conditional fixes in {total_time} seconds')

        self.stats.implied_nodes_time = total_time
        self.stats.num_implications_init = num_impl

        if self.problem.params.filter_implied_nodes:
            self.filter_implied_nodes()

    def filter_implied_nodes(self):
        """
        if u => v and v => w, then remove u => w
        """
        start_time = time.time()
        implied_by = {v: set() for v in self.graph.nodes}
        for w, i_set in self.implied_nodes.items():
            for v in i_set:
                implied_by[v].add(w)

        impl = sorted([(len(i_set), v, i_set) for v, i_set in implied_by.items() if i_set], reverse=True)

        for _, v, impl_set in impl:
            to_del = [(u, w) for w in self.implied_nodes[v] for u in impl_set if w in self.implied_nodes[u]]
            for u, w in to_del:
                self.implied_nodes[u].discard(w)
        num_impl = sum(len(v) for v in self.implied_nodes.values())
        logger.debug(f'Reduced to {num_impl} irreducible conditional fixes')

        self.stats.filter_implied_node_time = round(time.time() - start_time, 1)
        self.stats.num_implications_filtered = num_impl

        return

    #####################################################################
    def fap_naive(self):
        """
        just a sanity check
        """
        ap = []
        root = self.problem.root
        for u, v in itertools.combinations(self.graph.nodes, 2):
            if u == root or v == root:
                continue
            root_comp, _ = util.restricted_bfs(self.graph, root, [u, v])
            root_comp_weight = sum(self.bicolor_weights[n] for n in root_comp)

            if not self.check_weights(root_comp_weight):
                ap.append((u, v))

        return ap

    def find_activation_pairs(self):
        """
        At least one of u and v has to be activated, i.e.,
            u not in T => v in T

        Consider nodes v in DFS order.
            Consider nodes u of root component in G-v in DFS order.
                Consider root component C of G-v-u.
                If C feasible, do not go further in DFS u.
                else: add activation pair (u,v)
            If no activation pair was added:
            do not go further in DFS v.

        :return:
        """
        start_time = time.time()
        # ap = self.fap_naive()  # just for comparison

        root = self.problem.root

        # my DFS (adapted from nx)
        def add_neighbors_to_stack(u, v, stack):
            # add neighbors of v other than u to stack
            for w in self.graph[v]:
                if w != u:
                    stack.append((v, w))

        root_comp = {}
        visited = {root}
        checked = {root}  # to avoid symmetries and to gain speed-up
        stack = [(root, w) for w in self.graph[root]]
        while stack:
            u, v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            if v in self.fixed_nodes or v not in self.articulation_points:
                # we only consider articulation points that are not fixed
                add_neighbors_to_stack(u, v, stack)
                continue

            if not root_comp:
                root_comp = self._fix_nodes(virtual_removed_nodes=[v], visited=set(checked))
                checked.add(v)

            else:
                # add weight of u to root comp and see if some nodes are now possible
                root_comp = {k: val + self.bicolor_weights[u] for k, val in root_comp.items()
                             if not self.check_weights(val + self.bicolor_weights[u])}

            self.activation_pairs += [(v, x) for x in root_comp]

            if not root_comp:
                # no infeasibility => do not go further in DFS
                continue

            add_neighbors_to_stack(u, v, stack)

            if self.graph.degree(v) != 2:
                # reset root_comp
                root_comp = {}

        total_time = round(time.time() - start_time, 1)
        logger.debug(f'Found {len(self.activation_pairs)} activation pairs in {total_time} seconds')

    ####################################################################################################################
    # conditional removals

    def find_conflict_pairs(self):
        """
        Given that f is part of a solution, which nodes v cannot?
        We check for every pair f, v of vertices, if a minimum Steiner
        tree with terminals root, f, v is possible.
        Steiner tree with three terminals is easy with apsp, because
        the Steiner tree is the combination of three paths to a center
        node c (potentially also a terminal).

        TODO make this more efficient.
            - we can make use of implied nodes (filter dominated conflicts already here)
            - is there a clever way to exclude centers for a pair?
            - assume one node (e.g. root) as center and see, which conflict pairs can be excluded
            - cython?
        """
        start_time = time.time()
        conflict_pairs = set()

        # find bicolor labels from all (relevant) vertices
        #############################
        lengths = copy.deepcopy(self.bicolor_arc_weights)
        root = self.problem.root
        removed_lengths = [(v, root, np.array([0., 0.])) for v in self.graph[root]]

        bicolor_cutoff = self.get_bicolor_cutoff()
        # v -> dict of pareto-optimal labels to all nodes
        labels_from = {}

        # nodes of degree <= 2 cannot serve as centers for 3-terminal Steiner Tree
        # (exception: Steiner tree is a path, but then we can use a terminal)
        possible_centers = set(v for v, d in self.graph.degree() if d > 2).union({self.problem.root})

        # TODO exclude more centers?

        for v in possible_centers:

            # reinstate previously removed edges
            for u, w, l in removed_lengths:
                lengths[u][w] = l

            # remove new edges
            removed_lengths = []
            for w in self.graph[v]:
                v_len = lengths[w].pop(v)
                removed_lengths.append((w, v, v_len))

            labels = {w: [] for w in self.graph.nodes}
            labels[v].append((np.array((0., 0.)), set()))

            updated_labels = [(v, np.array((0., 0.)), set())]
            cutoff = bicolor_cutoff - v_len

            self._bicolor_radius(labels, lengths, updated_labels, cutoff=cutoff)

            labels_from[v] = {k: vl for k, vl in labels.items() if vl}
        logger.info(f'Computed bicolor steiner labels in {round(time.time() - start_time, 1)} seconds.')

        # find conflicts
        ####################
        # TODO restrict to nodes that do not imply other nodes
        # i.e.:  [v for v, vl in self.implied_nodes.items() if vl]
        # but also count how many conflict pairs one non-dominated
        # conflict dominates (to emphasize the importance of this
        # conflict).

        scoring_labels = self.get_scoring_1_labels()
        filter_method = self.problem.params.filter_conflicts
        pos_label_nodes = set(v for v, val in scoring_labels.items() if val >= self.threshold[filter_method])
        relevant_terminals = [v for v in pos_label_nodes if v != root]

        def check_center(u, v, c):
            """
            can c be the center of Steiner tree with terminals root, u, v?
            """
            if u not in labels_from[c] or v not in labels_from[c]:
                return False
            rc_labels = [l for l, _ in self.bicolor_labels[c]]
            ru_labels = [l for l, _ in self.bicolor_labels[u]]
            rv_labels = [l for l, _ in self.bicolor_labels[v]]
            cu_labels = [l for l, _ in labels_from[c][u]]
            cv_labels = [l for l, _ in labels_from[c][v]]

            cutoff_sum = self.problem.ub_length - self.bicolor_fixed_sum.sum()
            for l_cu, l_cv in itertools.product(cu_labels, cv_labels):
                l_cuv = l_cu + l_cv
                for l_rc in rc_labels:
                    l = l_rc + l_cuv
                    if all(l <= bicolor_cutoff) and sum(l) < cutoff_sum + EPSILON:
                        # there is a Steiner tree with terminals r, u, v with costs <= cutoff
                        return True

                for l_ru in ru_labels:
                    l = l_ru + l_cuv + self.bicolor_weights[c] - self.bicolor_weights[u]
                    if all(l <= bicolor_cutoff) and sum(l) < cutoff_sum + EPSILON:
                        # there is a Steiner tree with terminals r, v with costs <= cutoff
                        return True

                for l_rv in rv_labels:
                    l = l_rv + l_cuv + self.bicolor_weights[c] - self.bicolor_weights[v]
                    if all(l <= bicolor_cutoff) and sum(l) < cutoff_sum + EPSILON:
                        # there is a Steiner tree with terminals r, u with costs <= cutoff
                        return True

            return False

        def get_shortest_to_next_coarse_nodes(v):
            cv_labels = {}
            q = [(v, self.bicolor_weights[v])]
            visited = set()
            while q:
                element, label = q[0]
                del q[0]
                visited.add(element)
                for neighbor in self.graph[element]:
                    if self.graph.degree[neighbor] > 2:
                        cv_labels[neighbor] = label
                        continue
                    if neighbor in visited:
                        continue
                    q.append((neighbor, label + self.bicolor_weights[neighbor]))
            return cv_labels

        def check_fine_path_nodes(u, v):
            cu_labels = get_shortest_to_next_coarse_nodes(u)
            cv_labels = get_shortest_to_next_coarse_nodes(v)
            coarse_nodes = [c for c in cu_labels]
            rc_labels = []
            for c in coarse_nodes:
                rc_labels.append([l for l, _ in self.bicolor_labels[c]])

                # cuv_label = max(cu_labels[c], cv_labels[c])
                cuv_label = cu_labels[c] if (cu_labels[c] > cv_labels[c]).all() else cv_labels[c]
                for rc_label in rc_labels:
                    rcuv_label = rc_label + cuv_label
                    if (rcuv_label < bicolor_cutoff + EPSILON).all():
                        return True
                return False

        def check_terminals_fine_path(u, v):
            if self.graph.degree[u] > 2 or self.graph.degree[u] > 2:
                return False
            q = [u]
            visited = set()
            while q:
                element = q[0]
                del q[0]
                visited.add(element)
                for neighbor in self.graph[element]:
                    if self.graph.degree[neighbor] > 2 or self.problem.root == neighbor:
                        continue
                    if neighbor in visited:
                        continue
                    if neighbor == v:
                        return True
                    q.append(neighbor)
            return False

        for i, u in enumerate(relevant_terminals):
            # which centers have been useful for u, r?
            good_center_guesses = set()
            # TODO good_center_guesses as priority queue with pairs (node, num_successful_center)?
            # TODO or some other clever order in which we check centers
            for v in relevant_terminals[i + 1:]:

                conflict = True

                if check_terminals_fine_path(u, v):
                    if check_fine_path_nodes(u, v):
                        conflict = False
                else:
                    for c in good_center_guesses:
                        if check_center(u, v, c):
                            conflict = False
                            break

                    if conflict:
                        for c in possible_centers.difference(good_center_guesses):
                            if check_center(u, v, c):
                                conflict = False
                                good_center_guesses.add(c)
                                break

                if conflict:
                    conflict_pairs.add((u, v))

        run_time = round(time.time() - start_time, 1)
        self.stats.conflict_pair_time = run_time
        self.stats.num_conflicts_init = len(conflict_pairs)
        logger.debug(f'Found {len(conflict_pairs)} conflict pairs after {run_time} seconds.')
        # WARNING: Wenn kein anderes center fuer die Terminals passt, dann muessen wir auch Terminals,
        # die keine possibel_centers sind checken, um den Pfadfall abzusichern.

        # import sys
        # sys.exit()

        # self.debug_plot()

        return conflict_pairs

    def find_forbidden_pairs_old(self):
        """
        Given that f is part of a solution, which nodes v cannot?
        We check for every pair f, v of vertices, if a minimum Steiner
        tree with terminals root, f, v is possible.
        Steiner tree with three terminals is easy with apsp, because
        the Steiner tree is the combination of three paths to a center
        node c (potentially also a terminal).

        TODO make this more efficient.
            - we can make use of implied nodes (filter dominated conflicts already here)
            - is there a clever way to exclude centers for a pair?
            - cython?
        """
        start_time = time.time()

        digraph = self.graph.to_directed()
        aux_weights = {
            0: ('b_weight', {(u, v): self.bicolor_weights[v][0] for u, v in digraph.edges}),
            1: ('r_weight', {(u, v): self.bicolor_weights[v][1] for u, v in digraph.edges}),
        }
        if self.problem.params.find_conflict_pairs_mode == 2:
            aux_weights[2] = ('weight', {(u, v): self.graph.nodes[v]['length'] for u, v in digraph.edges})

        forbidden_pairs = set()
        r = self.problem.root

        scoring_labels = self.get_scoring_1_labels()
        filter_method = self.problem.params.filter_conflicts
        pos_label_nodes = set(v for v, val in scoring_labels.items() if val >= self.threshold[filter_method])

        for color, pair in aux_weights.items():
            if color != 2:
                cutoff = self.weight_ranges[1][color]
            else:
                cutoff = self.problem.ub_length
            nx.set_edge_attributes(digraph, pair[1], pair[0])
            apsp = dict(nx.all_pairs_dijkstra_path_length(digraph, weight=pair[0], cutoff=cutoff))

            non_root_nodes = [v for v in pos_label_nodes if v != r]
            for f, v in itertools.combinations(non_root_nodes, 2):
                if (f, v) in forbidden_pairs:
                    # pair is already ruled out
                    # TODO is this faster? start with aux_weights[0] if possible?
                    continue

                # if self.problem.params.only_pos_forbidden_pairs:
                #     if self.graph.nodes[f]['profit'] < 0 or self.graph.nodes[v]['profit'] < 0:
                #         continue

                if v not in apsp[f]:
                    # shortest path is too long, no need for Steiner tree
                    forbidden_pairs.add((f, v))
                    continue
                feas = False
                debug_weight = 100000000000
                for c, w in apsp[r].items():
                    weight = w + apsp[c].get(f, cutoff) + apsp[c].get(v, cutoff)
                    debug_weight = min(debug_weight, weight)
                    if weight < cutoff + EPSILON:
                        # there is a Steiner tree with terminals r, f, v with costs <= cutoff
                        feas = True
                        break
                if not feas:
                    forbidden_pairs.add((f, v))

        self.forbidden_pairs = forbidden_pairs
        run_time = round(time.time() - start_time, 1)
        logger.debug(f'Found {len(forbidden_pairs)} forbidden pairs after {run_time} seconds.')
        self.stats.conflict_pair_time = run_time
        self.stats.num_conflicts_init = len(forbidden_pairs)

        return forbidden_pairs

    def filter_conflict_pairs(self):

        filter_method = self.problem.params.filter_conflicts

        filter_function = {
            0: None,  # no conflict filter
            # 1: self.filter_conflicts_scoring_1,
            1: None,
            2: self.filter_conflicts_dominate,
            3: self.filter_conflicts_dominate,
            4: self.filter_conflicts_dominate,
            5: self.filter_conflicts_dominate,
            6: self.filter_conflicts_dominate,
            7: self.filter_conflicts_dominate,
            8: self.filter_conflicts_dominate,
        }

        if filter_function[filter_method] is not None:
            filter_function[filter_method](self.threshold[filter_method])
        logger.info(f'Identified {len(self.forbidden_pairs)} essential conflict pairs.')
        self.stats.num_conflicts_filtered = len(self.forbidden_pairs)
        return

    def get_scoring_1_labels(self):
        graph = self.graph
        weight_func = lambda u, v, x: graph.nodes[v]['length']

        spl = dict(nx.single_source_dijkstra_path_length(
            graph, self.problem.root, cutoff=self.problem.ub_length,
            weight=weight_func
        ))

        # transform lengths such that (lb_length, 0) -> (0, 1)
        dist_labels = {v: max(-1, (1 - (spl[v] / (self.problem.lb_length * .75)))) for v in spl}
        # ksplot.draw(graph, node_color_dict=dist_labels)

        # transform profits such that (0, max) -> (0, 1)
        profits = dict(graph.nodes(data='profit'))
        # min_profit, max_profit = min(profits.values()), max(profits.values())
        # profit_labels = {v: (p - min_profit) * 2 / (max_profit - min_profit) - 1 for v, p in profits.items()}
        s_profits = sorted((p, v) for v, p in graph.nodes(data='profit'))
        max_profit = s_profits[-1][0]
        profit_labels = {v: p / max_profit for v, p in profits.items()}
        # ksplot.draw(graph, node_color_dict=profit_labels)

        # top 5% profits get a boost
        top_profits = s_profits[-int(.05 * len(profits)):]
        for p, v in top_profits:
            profit_labels[v] += 1

        # between -1 and 1
        comb_labels = {v: max(-1, min(1, dist_labels[v] + 0.5 * profit_labels[v])) for v in graph.nodes}
        return comb_labels

    def filter_conflicts_scoring_1(self, threshold=0):
        """
        Used for INOC paper. (is that correct?)
        TODO

        """

        comb_labels = self.get_scoring_1_labels()

        pos_label_nodes = set(v for v, val in comb_labels.items() if val >= threshold)
        pos_label_cp = [(u, v) for u, v in self.forbidden_pairs if u in pos_label_nodes and v in pos_label_nodes]

        self.forbidden_pairs = pos_label_cp

    def filter_conflicts_dominate(self, threshold=0):
        """
        conflicts uw, vw and implied node u => v, then uw is dominated by vw
        keep only non-dominated conflicts

        TODO not really tested

        TODO we can do this faster, and even within finding conflict pairs.
        But at this point it is not so important to find essential conflicts fast,
        but to find good essential conflicts. Only if this is successful,
        we should optimize the variant.

        TODO maybe combine dominated conflicts with another filtering. To
        this end it could be interesting to know how many other conflict
        pairs are dominated by one remaining pair (this pair should probably
        be considered).

        """
        # v -> nodes that are in conflict with node v
        forbidden = {v: set() for v in self.graph.nodes}
        for u, v in self.forbidden_pairs:
            forbidden[u].add(v)
            forbidden[v].add(u)

        # v -> nodes that are "further away" than v
        implied_by = {v: set() for v in self.graph.nodes}
        for w, i_set in self.implied_nodes.items():
            for v in i_set:
                implied_by[v].add(w)

        # sort to start with conflicts dominating many others
        impl = sorted([(len(i_set), v, i_set) for v, i_set in implied_by.items() if i_set], reverse=True)

        non_dominated_conflicts = set()
        dominates = {}

        scoring_labels = self.get_scoring_1_labels()
        non_dominated_good_conflicts = set()
        dominates_good_conflicts = {}

        for _, v, v_i_set in impl:
            # sort to start with conflicts dominating many others
            c_list = sorted(forbidden[v], key=lambda w: len(implied_by[w]), reverse=True)
            for c in c_list:
                if c not in forbidden[v]:
                    # already handled
                    continue
                # c is in conflict with v and now we consider vertices implied by c
                # (in fact, these should be all in conflict with v)
                key = tuple(sorted([v, c]))
                dominates.setdefault(key, set())
                dominates_good_conflicts = False

                v_implied_list = [v] + list(v_i_set)
                c_implied_list = list(implied_by[c]) + [c]
                for x, y in itertools.product(v_implied_list, c_implied_list):
                    if y in forbidden[x] or x in forbidden[y]:
                        forbidden[x].discard(y)
                        forbidden[y].discard(x)
                        val = tuple(sorted([x, y]))
                        dominates[key].add(val)
                        if scoring_labels[x] > threshold and scoring_labels[y] > threshold:
                            dominates_good_conflicts = True
                non_dominated_conflicts.add(key)

                if dominates_good_conflicts:  # a conflict pair dominates itself
                    non_dominated_good_conflicts.add(key)

        # TODO It could be better to keep
        # self.forbidden_pairs = non_dominated_conflicts
        #
        # best_dominations = sorted([(len(ds), p) for p, ds in dominates_good_conflicts.items()], reverse=True)
        self.forbidden_pairs = non_dominated_good_conflicts

        # for u, v in self.forbidden_pairs:
        #     print(u, v)
        #     self.debug_plot(nodes=[u,v])

        return dominates

    ####################################################################################################################
    # rings around the root

    def compute_root_rings(self):

        color_lb = self.weight_ranges[0]

        comp_weight = self.bicolor_weights[self.problem.root].copy()

        prev_nodes = set()
        curr_nodes = {self.problem.root}
        while any(comp_weight < color_lb):
            arc_cut = []
            for v in curr_nodes:
                for w in self.graph.neighbors(v):
                    if w in curr_nodes or w in prev_nodes:
                        continue
                    arc_cut.append((v, w))

            prev_nodes, curr_nodes = curr_nodes, set(a[1] for a in arc_cut)
            comp_weight += sum(self.bicolor_weights[v] for v in curr_nodes)
            self.root_rings.append(arc_cut)

        return

    def compute_lb_disk(self):
        """
        compute nodes that are within lb distance to the root.
        :return:
        """
        graph = self.graph
        weight_func = lambda u, v, x: graph.nodes[v]['length']

        spl = dict(nx.single_source_dijkstra_path_length(
            graph, self.problem.root, cutoff=self.problem.lb_length,
            weight=weight_func
        ))
        self.lb_disk_nodes = list(spl.keys())

    ####################################################################################################################
    # general stuff

    def debug_plot(self, nodes=None):
        if nodes is None:
            nodes = []
        g2 = nx.to_undirected(self.graph)
        # ksplot.draw_bwr(g2)
        node_colors = dict(self.graph.nodes(data='color'))
        draw.draw(g2, root=self.problem.root, node_color_dict=node_colors, colormap='bwr', highlight_node_list=nodes)


def plot_dominating_conflicts(pp):
    old_conflict_pairs = pp.find_forbidden_pairs_old()
    pp.forbidden_pairs = old_conflict_pairs
    old_dominate = pp.filter_conflicts_dominate()
    old_dominate_list = list(set([conflict_pair for key in old_dominate for conflict_pair in old_dominate[key]]))
    old_non_dominate_list = list(set([conflict_pair for conflict_pair in old_conflict_pairs
                                      if conflict_pair not in old_dominate_list]))

    new_conflict_pairs = pp.find_conflict_pairs()
    pp.forbidden_pairs = new_conflict_pairs
    new_dominate = pp.filter_conflicts_dominate()
    new_dominate_list = list(set([conflict_pair for key in new_dominate for conflict_pair in new_dominate[key]]))
    new_non_dominate_list = list(set([conflict_pair for conflict_pair in new_conflict_pairs
                                      if conflict_pair not in new_dominate_list]))

    new_old_dominate = [conflict_pair for conflict_pair in new_dominate if conflict_pair not in old_dominate_list
                        and conflict_pair in old_conflict_pairs]
    # new_old_dominate = [conflict_pair for conflict_pair in new_dominate if conflict_pair not in old_dominate_list]
    print(len(old_non_dominate_list))
    print(len(new_non_dominate_list))

    # ksplot.draw_old_and_new_conflict_pairs(self.graph, self.problem.root, old_conflict_pairs, new_conflict_pairs)
    # ksplot.draw_old_and_new_conflict_pairs(self.graph, self.problem.root, old_dominate_list, new_old_dominate)
    ksplot.draw_old_and_new_conflict_pairs(pp.graph, pp.problem.root, old_non_dominate_list, new_non_dominate_list)

    # ---- plot plain conflict pairs new and old ----
    # edges = list(new_conflict_graph.edges())
    # new_conflict_graph.remove_edges_from(edges)
    # for u, v in new_conflict_pairs:
    #     new_conflict_graph.add_edge(u, v)
    # new_conflict_graph = new_conflict_graph.to_undirected()
    # ksplot.draw_node_weighted_graph2(new_conflict_graph, root=self.problem.root, figure=1)
    # old_conflict_pairs = pp.find_forbidden_pairs_old()
    # old_conflict_graph = pp.graph.copy()
    # edges = list(old_conflict_graph.edges())
    # old_conflict_graph.remove_edges_from(edges)
    # for u, v in old_conflict_pairs:
    #     old_conflict_graph.add_edge(u, v)
    # old_conflict_graph = old_conflict_graph.to_undirected()
    # ksplot.draw_node_weighted_graph2(old_conflict_graph, root=pp.problem.root, figure=2)
