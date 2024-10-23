import itertools
import logging
import sys

gurobipath = '/nfs/optimi/usr/sw/gurobi952/linux64/lib/python3.8_utf32'
if gurobipath not in sys.path:
    sys.path.append(gurobipath)

import gurobipy
import networkx as nx

import brcmwcs.draw as draw
import brcmwcs.preprocessing as prep
import brcmwcs.separation as separation
import brcmwcs.utilities as util

import brcmwcs.ConflictGraph as cg

logger = logging.getLogger()

EPSILON = 1e-6


class OptModel:

    def __init__(self, problem, print_gurobi=False):

        self.problem = problem

        ##########

        self.graph = problem.graph.to_directed()
        self.params = problem.params
        self.root = problem.root
        self.delta = problem.delta
        self.lb_length = problem.lb_length
        self.ub_length = problem.ub_length

        self.model = gurobipy.Model()
        self.variables = {}

        # output printed=
        self.model.Params.OutputFlag = int(print_gurobi)
        # self.model.Params.LogToConsole = 0

        # core graph is the graph with recursively removed nodes of degree 1
        self.core_graph = None

        # does the core graph contain more than the root (<=> graph is not a tree)
        self.proper_core_graph = True

        # for potential coarse-to-fine option
        self.coarse_graph = None

        # graph on which connectivity is enforced (for c2f option)
        self.opt_graph = None

    def setup(self, set_objective=False):

        self.model._callback_fun = None
        self.model.ModelSense = gurobipy.GRB.MAXIMIZE

        y = self.model.addVars(self.graph.nodes, vtype='B', name='y')
        # y = self.model.addVars(self.graph.nodes, lb=0, ub=1, vtype='C', name='y')
        self.variables['y'] = y

        # set objective (if possible)
        if set_objective:
            profit = dict(self.graph.nodes(data='profit'))
            self.set_objective(profit, cutoff=None)

        # root is active
        self.model.addLConstr(y[self.root] == 1)

        # length constraints
        length = dict(self.graph.nodes(data='length'))
        len_sum = sum([length[v] * y[v] for v in y])
        self.model.addLConstr(len_sum >= self.lb_length - EPSILON, "lb_length_constr")
        self.model.addLConstr(len_sum <= self.ub_length + EPSILON, "ub_length_constr")

        ###################
        # color constraints

        if self.params.add_balance_constr or self.params.add_color_range_constr:
            red_nodes = [v for v in self.graph if self.graph.nodes[v]['color'] == 1]
            red_weight = sum([length[v] * y[v] for v in red_nodes])

            blue_nodes = [v for v in self.graph if self.graph.nodes[v]['color'] == -1]
            blue_weight = sum([length[v] * y[v] for v in blue_nodes])

        # balance constraint
        if self.params.add_balance_constr:
            self.model.addLConstr(red_weight - blue_weight <= self.delta + EPSILON, "balance_constr_1")
            self.model.addLConstr(blue_weight - red_weight <= self.delta + EPSILON, "balance_constr_2")

        ##########################
        # connectivity constraints

        # restrict connectivity to core graph
        if not hasattr(self.problem, 'core_graph'):
            # preprocessing was not run => only find core graph
            preprocessing = prep.Preprocessing(self.problem)
            preprocessing.find_network_induced_cuts()
            preprocessing.build_core_graph()
            self.problem.prep_out = prep.PrepOutput(preprocessing)

        self.core_graph = self.problem.core_graph
        self.proper_core_graph = self.core_graph.number_of_nodes() > 1

        self.opt_graph = self.core_graph.to_directed()  # is overwritten for c2f

        if self.proper_core_graph:
            # if the core graph has only the root, all connectivity is enforced by implied nodes

            # warmstart
            if self.params.use_warmstart_heur:
                self.set_heuristic_warmstart()

            connectivity_setup_fct = {
                'scf': self.setup_scf_conn,
                'mcf': self.setup_mcf_conn,
                'arc_sep': self.setup_arc_sep_conn,
                'node_sep': self.setup_node_sep_conn,
                'c2f_scf': self.setup_c2f_scf_conn,
                'c2f_arc_sep': self.setup_c2f_arc_sep_conn,
                'hybrid': self.setup_hybrid_conn,
                'tree_heuristic': self.setup_tree_heuristic_conn
            }

            # call correct connectivity setup function
            connectivity_setup_fct[self.params.connectivity_mode]()

        prep_out = self.problem.prep_out

        ####################
        # preprocessing info

        if self.params.add_color_range_constr and prep_out.color_range is not None:
            self.model.addLConstr(blue_weight >= prep_out.color_range[-1][0] - EPSILON)
            self.model.addLConstr(blue_weight <= prep_out.color_range[-1][1] + EPSILON)
            self.model.addLConstr(red_weight >= prep_out.color_range[1][0] - EPSILON)
            self.model.addLConstr(red_weight <= prep_out.color_range[1][1] + EPSILON)

        # fixed nodes
        self.model.addConstrs((y[v] == 1 for v in prep_out.fixed_nodes if v in y), name='fixed')

        # implied nodes
        for v, impl_list in prep_out.implied_nodes.items():
            if impl_list:
                self.model.addConstrs((y[v] <= y[u] for u in impl_list), name='implied')

        # activation pairs
        self.model.addConstrs((y[u] + y[v] >= 1 for u, v in getattr(prep_out, 'activation_pairs', [])),
                              name='activation')

        if not self.proper_core_graph:
            return

        # the following is only if there are z variables

        # extended arc neighborhood
        if self.params.arc_neighborhood_cuts:
            self.add_arc_neighborhood_cuts(coarse_depth=self.params.arc_neighborhood_cuts)

        # if True:
        #     self.add_k_cycle_inequalities(k=3)

        if self.params.add_indegree_constraints:
            self.add_indegree_constraints(nodes=prep_out.lb_disk_nodes)

        if self.params.root_ring_cuts:
            self.add_root_ring_cuts(prep_out.root_rings)

        if self.params.fine_path_cuts:
            target_nodes = [p[0] for p in sorted(self.opt_graph.nodes(data='profit'), key=lambda x: x[1])[-10:]]
            self.add_finepath_cuts(target_nodes, self.params.fine_path_cuts)

        # ##########################
        # if params.cb_cuts == 3:
        #     # lp plots
        #     self.model._callback_fun = cb_plot_lp_relaxation
        #     self.model._myvars = self.variables
        #     self.model._graph = self.problem.graph
        #     self.model._root = self.problem.root
        #     self.model._cnt = 0
        # elif params.cb_cuts:
        #     # self.model.Params.PreCrush = 1
        #     self.model._callback_fun = cb_forbidden_pair_constraints
        #     # give new attributes to model to use those in the callback
        #     self.model._cb_param = 0  # params.cb_cuts
        #     self.model._myvars = self.variables
        #     self.model._forbidden_pairs = {p: 0 for p in prep_out.forbidden_pairs}
        #     self.model._cb_cut_cnt = 0
        #     # for plotting:
        #     self.model._graph = self.problem.graph
        #
        # if params.knapsack_constraints:
        #     disk_centers_rhs_list = self.find_disk_centers()
        #     for disk_centers, rhs in disk_centers_rhs_list:
        #         self.add_knapsack_constraints(disk_centers, rhs)
        #         self.add_knapsack_constraints_advanced(disk_centers, rhs)
        #

        if self.params.babo_cliques:
            conflict_graph = cg.ConflictGraph(self.problem, prep_out.forbidden_pairs)
            cliques = conflict_graph.find_clique_covering(self.problem)
            for i, clique in enumerate(cliques):
                self.model.addLConstr(sum(y[v] for v in clique) <= 1, f"clique_{i}")

        elif self.params.find_conflict_pairs_mode:
            logger.info(f'Considering {len(prep_out.forbidden_pairs)} conflict pairs.')
            for u, v in prep_out.forbidden_pairs:
                self.model.addLConstr(y[u] + y[v] <= 1, f"forbidden_{u}_{v}")

    ####################################################################################################################
    # LP strengthening
    def add_k_cycle_inequalities(self, k=3):
        """
        identify cycles of length k in [3] without root
        add constraints:
         sum(z in cycle) <= sum(y in cycle minues 1 node)  (i.e. k constrs)


        :param k:
        :return:
        """
        if 'z' not in self.variables:
            return

        opt_graph = self.opt_graph

        if self.coarse_graph is not None:
            rel_arcs = set(tuple(sorted(a)) for a in self.coarse_graph.edges())
        else:
            coarse_nodes = set(v for v, d in self.opt_graph.in_degree if d > 2)
            rel_arcs = set(tuple(sorted((u, v))) for v in coarse_nodes for u in opt_graph[v])

        # find triangles
        triangles = []

        while rel_arcs:
            u, v = rel_arcs.pop()
            common_neighbors = [w for w in opt_graph[u] if w in opt_graph[v]]
            for w in common_neighbors:
                rel_arcs.discard(tuple(sorted([u, w])))
                rel_arcs.discard(tuple(sorted([v, w])))
                triangles.append([u, v, w])

        # add inequalities
        y = self.variables['y']
        z = self.variables['z']

        for u, v, w in triangles:
            sum_z = z[u, v] + z[v, w] + z[w, u] + z[u, w] + z[w, v] + z[v, u]

            for pair in itertools.combinations([u, v, w], 2):
                self.model.addLConstr(sum_z <= sum(y[n] for n in pair), f'cycle_cut_{pair}')

        return

    def add_arc_neighborhood_cuts(self, coarse_depth=2):
        """
        For each node, we add a constraint, that an ingoing arc has to be active
        if the node is active.

        For coarse nodes, we do this also with ingoing arcs to neighbors (i.e., depth 2).
        """
        if 'z' not in self.variables:
            return

        opt_graph = self.opt_graph

        y = self.variables['y']
        z = self.variables['z']

        # indegree constraints have already been added
        #
        # # if y is active, an ingoing (binary) arc has to be active (indegree)
        # for v in opt_graph.nodes:
        #     if self.params.connectivity_mode in ['arc_sep', 'c2f_arc_sep']:
        #         # these constraints have already been added for arc separation
        #         break
        #     if v == self.root:
        #         continue
        #     if isinstance(opt_graph, nx.MultiDiGraph):
        #         sum_in = sum(z[a] for a in opt_graph.in_edges(v, keys=True))
        #     else:
        #         sum_in = sum(z[a] for a in opt_graph.in_edges(v))
        #
        #     self.model.addLConstr(sum_in == y[v], f'arc_cut_{v}')

        coarse_nodes = set(v for v, d in self.opt_graph.in_degree if d > 2) if self.coarse_graph is None \
            else set(self.coarse_graph.nodes)
        coarse_nodes.discard(self.root)

        # removing articulation points can be bad for triangle with branches.
        # All nodes are articulation points, but we still want to avoid the cycle.
        # coarse_nodes.difference_update(self.problem.graph.articulation_points)

        for v in coarse_nodes:
            visited = {v}
            curr_node_cut = list(opt_graph.neighbors(v))
            for j in range(coarse_depth - 1):
                if self.root in curr_node_cut:
                    break
                curr_arc_cut = []

                if isinstance(opt_graph, nx.MultiDiGraph):
                    rel_arcs = opt_graph.in_edges(curr_node_cut, keys=True)
                else:
                    rel_arcs = opt_graph.in_edges(curr_node_cut)

                for a in rel_arcs:
                    if a[0] in visited or a[0] in curr_node_cut:
                        continue
                    curr_arc_cut.append(a)

                sum_in = sum(z[a] for a in curr_arc_cut)
                self.model.addLConstr(sum_in >= y[v], f'in_arc_cut_{v}_{j + 1}')

                visited.update(curr_node_cut)

                # new node cut is tails from curr_arc_cut
                curr_node_cut = set(c[0] for c in curr_arc_cut)
                # draw.draw(flow_graph, root=self.root, highlight_edge_list=curr_arc_cut, highlight_node_list=[v])
        return

    def add_root_ring_cuts(self, root_rings):
        """
        One arc of every arc set in distance (1,2,3,..) has to be active, until the union
        contains a feasible subgraph. These are computed during preprocessing and given
        to the function.
        """

        if 'z' not in self.variables:
            return

        logger.debug(f'Adding root ring cuts for {len(root_rings)} rings..')

        # if c2f with fine flow, we have to use the y variables, because fine arcs have no z variables
        if self.params.connectivity_mode in ['c2f_scf', 'c2f_arc_sep', 'hybrid'] \
                and self.params.fine_path_model == 'flow':
            y = self.variables['y']
            for i, ring in enumerate(root_rings):
                self.model.addLConstr(
                    sum(y[a[1]] for a in ring) >=1, name=f'root_ring_{i}'
                )
            return

        y = self.variables['y']
        z = self.variables['z']
        zf = self.variables.get('zf', {})

        # map_missing_arcs (fine arcs fine node -> coarse node)
        if self.coarse_graph is not None:
            missing_arc_map = {(p[-1], a[1]): a for a, p in self.coarse_graph.proper_coarse_arcs}
        else:
            missing_arc_map = {}

        for i, ring in enumerate(root_rings):

            if isinstance(self.opt_graph, nx.MultiDiGraph):
                ring_z_vars = [(u, v, i) for u, v, i in z if (u, v) in ring]
            else:
                ring_z_vars = [a for a in z if a in ring]

            ring_zf_vars = [a for a in zf if a in ring]

            ring_y_vars = []

            # check if there are missing arcs
            missing_arcs = [(u, v) for u, v in ring
                            if (u, v) not in ring_zf_vars and (u, v) not in [t[:2] for t in ring_z_vars]]
            for u, v in missing_arcs:
                if (u, v) in missing_arc_map:
                    a = missing_arc_map[(u, v)]  # this is a coarse arc
                    # instead of the last fine arc (which has no variable),
                    # we activate the corresponding coarse arc
                    ring_z_vars.append(a)
                else:
                    # the arc was cut away when the core graph was built
                    # therefore, we use the y variable of the head
                    ring_y_vars.append(v)

            self.model.addLConstr(
                sum(z[a] for a in ring_z_vars) + sum(zf[a] for a in ring_zf_vars) + sum(y[v] for v in ring_y_vars) >= 1,
                name=f'root_ring_{i}'
            )

        # # draw root rings
        # edge_colors = {k: v for d in [{e: i+1 for e in ring} for i, ring in enumerate(root_rings)]
        #                for k, v in d.items()}
        # aux_graph = nx.Graph(self.opt_graph)
        # draw.draw(aux_graph, root=self.root, edge_color_dict=edge_colors, colormap='jet', node_list=[])

    def add_indegree_constraints(self, nodes=None):
        """
         z(v,w) <= sum_(v neq w) z(u,v)
         These are implied by indegree and 2-cycle cuts...
        """
        graph = self.opt_graph
        z = self.variables['z']

        if nodes is None:
            nodes = graph.nodes

        for v in nodes:
            if v == self.problem.root or v not in graph.nodes:
                continue

            if isinstance(graph, nx.MultiDiGraph):
                in_arcs = list(graph.in_edges(v, keys=True))
                out_arcs = list(graph.out_edges(v, keys=True))
            else:
                in_arcs = list(graph.in_edges(v))
                out_arcs = list(graph.out_edges(v))

            for oa in out_arcs:
                rel_sum = sum(z[ia] for ia in in_arcs if ia[0] != oa[1])
                self.model.addLConstr(rel_sum >= z[oa], name=f'indegree_{oa}')

    def add_finepath_cuts(self, target_nodes, variant=2):
        """
        For a given set of "important" nodes, we compute a min-cut in the coarse graph.
        The edge capacity of coarse edge depends on the  variant

        3: the cumulated profit of this edge, multiplied by -1 + maxprofit (to get cap >=0)
        2: the cumulated weight of this edge. More specifically, w' = avg cumulated weight, then
            cap = 1/ (w / w')^2
        1: number k of fine nodes within this edge. More specifically, the capacity is set to
                    1 / (k+1)^2

        A cover constraint for the z variables of the cut arcs is then added.
        """
        if 'z' not in self.variables:
            return

        y = self.variables['y']
        z = self.variables['z']

        if self.coarse_graph is not None:
            min_cut_graph = nx.Graph(self.coarse_graph)
        else:
            coarse_nodes = set(v for v, d in self.core_graph.degree if d > 2).union([self.root])
            # coarse_nodes = set(v for v, d in self.core_graph.degree if d > 2).union([self.root]).union(target_nodes)
            coarse_graph = util.build_coarse_graph(self.core_graph, coarse_nodes, allow_multiarcs=False)
            min_cut_graph = nx.Graph(coarse_graph)

        if variant == 1:
            for u, v, p in min_cut_graph.edges(data='interior_fine_path'):
                cap = 1 / (len(p) + 1) ** 2
                min_cut_graph[u][v]['capacity'] = cap

        elif variant == 2:
            length = dict(self.graph.nodes(data='length'))
            ld = {(u, v): length[u] + sum(length[n] for n in p) + length[v]
                  for u, v, p in min_cut_graph.edges(data='interior_fine_path')}
            avg_l = sum(ld.values()) / len(ld)
            transformed_l = {a: (avg_l/l)**2 for a, l in ld.items()}
            nx.set_edge_attributes(min_cut_graph, transformed_l, 'capacity')

            # for u, v, p in min_cut_graph.edges(data='interior_fine_path'):
            #     sum_len = length[u] + sum(length[n] for n in p) + length[v]
            #     min_cut_graph[u][v]['capacity'] = 1/sum_len ** 2

        elif variant == 3:

            profit = dict(self.graph.nodes(data='profit'))
            max_profit = max(profit.values())
            shifted_profit = {v: -p + max_profit for v, p in profit.items()}
            for u, v, p in min_cut_graph.edges(data='interior_fine_path'):
                cap = shifted_profit[u] + sum(shifted_profit[n] for n in p) + shifted_profit[v]
                min_cut_graph[u][v]['capacity'] = cap

        elif variant == 4:
            # like variant 2 but with additional cuts z_uv >= z_vw for fine path triple nodes u, v, w
            length = dict(self.graph.nodes(data='length'))
            ld = {(u, v): length[u] + sum(length[n] for n in p) + length[v] for u, v, p in
                  min_cut_graph.edges(data='interior_fine_path')}
            avg_l = sum(ld.values()) / len(ld)
            transformed_l = {a: (avg_l / l) ** 2 for a, l in ld.items()}
            nx.set_edge_attributes(min_cut_graph, transformed_l, 'capacity')

            num_ifp = 0
            for u, v, p in coarse_graph.edges(data='interior_fine_path'):
                if not p:
                    continue
                p_plus = [u] + p + [v]
                for i in range(1, len(p_plus)-1):
                    a, b, c = p_plus[i-1:i+2]
                    self.model.addLConstr(z[a, b] >= z[b, c], name=f'fine_path_{a,b,c}')
                    num_ifp += 1
                    # self.model.addLConstr(z[c, b] >= z[b, a], name=f'fine_path_{c,b,a}')

            logger.info(f'\n\n{num_ifp}\n')

        elif variant == 5:
            # like variant 2 but with additional cuts z_uv >= z_vw for fine path triple nodes u, v, w
            length = dict(self.graph.nodes(data='length'))
            ld = {(u, v): length[u] + sum(length[n] for n in p) + length[v] for u, v, p in
                  min_cut_graph.edges(data='interior_fine_path')}
            avg_l = sum(ld.values()) / len(ld)
            transformed_l = {a: (avg_l / l) ** 2 for a, l in ld.items()}
            nx.set_edge_attributes(min_cut_graph, transformed_l, 'capacity')

        ##############################
        # add cuts
        fine_path_cut_triples = set()
        for v in target_nodes:
            if v not in min_cut_graph or v == self.root:
                continue

            cut_value, partition = nx.minimum_cut(min_cut_graph, self.root, v)
            reachable, non_reachable = partition

            cutset = set()
            for u, nbrs in ((n, min_cut_graph[n]) for n in reachable):
                cutset.update((u, v) for v in nbrs if v in non_reachable)

            # draw.draw(min_cut_graph, root=self.root, highlight_edge_list=cutset, highlight_node_list=[v])

            # ######################
            # # draw
            # h_nodes = set(n for s in [set([u, v]).union(coarse_graph.edges[u, v]['interior_fine_path'])
            #                           for u, v in cutset]
            #               for n in s)
            # h_edges = [(u, v) for u, v in self.problem.graph.edges if u in h_nodes and v in h_nodes]
            # node_colors = {v: 1 for v in h_nodes}
            # draw.draw(self.problem.graph, root=self.root, node_color_dict=node_colors, node_list=h_nodes,
            #           highlight_edge_list=h_edges, highlight_node_list=[v], colormap='bwr')
            # #######################

            # separating_z_vars = [(u, v, i) for u, v, i in z if (u, v) in cutset]

            if self.coarse_graph is not None:
                if isinstance(self.coarse_graph, nx.MultiDiGraph):
                    cut_var_arcs = [(u, v, i) for u, v, i in z if (u, v) in cutset]
                else:
                    cut_var_arcs = cutset
            else:
                cut_var_arcs = [(u, v) if (u, v) in z else (coarse_graph[u][v]['interior_fine_path'][-1], v)
                                for u, v in cutset]

                # if variant =5, add fine_pred_arc_cuts for cut arcs

                if variant == 5:
                    for u, v in cutset:
                        p = coarse_graph.edges[u, v]['interior_fine_path']
                        if not p:
                            continue
                        p_plus = [u] + p + [v]
                        for i in range(1, len(p_plus) - 1):
                            fine_path_cut_triples.add(tuple(p_plus[i - 1:i + 2]))
                            # a, b, c = p_plus[i - 1:i + 2]
                            # self.model.addLConstr(z[a, b] >= z[b, c], name=f'fine_path_{a, b, c}')

            self.model.addLConstr(
                sum(z[a] for a in cut_var_arcs) >= y[v],
                name=f'fine_path_cut_{v}'
            )

        for a, b, c in fine_path_cut_triples:
            self.model.addLConstr(z[a, b] >= z[b, c], name=f'fine_path_{a, b, c}')

        return

    ####################################################################################################################
    def set_objective(self, weights, cutoff=None):
        """
        Bring current duals in the objective and set cutoff.
        :param weights: dict node -> node weight, already transformed for maximization
        :param cutoff: we do not want solutions with worse objective.
                        This can also be given to terminate LRMWCS earlier
                        for single center nodes (e.g., if the obj of the best
                        solution of the pricing problem so far is given).
        """
        for v, var in self.variables['y'].items():
            var.obj = weights[v]

        if cutoff is not None:
            self.model.params.cutoff = cutoff

    def optimize(self):

        if self.model._callback_fun is None:
            self.model.optimize()
        else:
            self.model.optimize(self.model._callback_fun)

    def get_solution(self, solution=None):
        # #########################
        # ## draw LP Relax
        # y = self.variables['y']
        # import brcmwcs.draw as draw
        # draw_g = nx.to_undirected(self.problem.graph)
        # non_zero_nodes = [v for v, val in y.items() if val.x > EPSILON and v in draw_g]
        # node_colors = {v: val.x for v, val in y.items() if val.x > EPSILON and v in draw_g}
        # draw.draw(draw_g, root=self.root, node_list=non_zero_nodes, node_color_dict=node_colors, colormap='OrRd')
        # ########################

        # retrieve solution
        sol_nodes = [v for v, v_var in self.variables['y'].items() if v_var.x > 0.5]

        if solution is None:
            return sol_nodes

        solution.subgraph_nodes = sol_nodes
        solution.stats['opt_time'] = round(self.model.runtime, 2)
        solution.stats['obj_val'] = round(self.model.objval, 4)

        return solution

    ####################################################################################################################
    ####################################################################################################################
    # connectivity

    ###########################################################
    # scf
    def setup_scf_conn(self, flow_graph=None, c2f=False):
        """
        :param flow_graph: DiGraph or MultiDiGraph (if used for c2f)
        :param c2f: part of coarse-to-fine model?
        """
        if flow_graph is None:
            flow_graph = self.opt_graph
        else:
            self.opt_graph = flow_graph

        if c2f:
            big_M = util.compute_c2f_big_m(self.core_graph, flow_graph, self.root, self.ub_length)
        else:
            big_M = util.compute_scf_big_m(flow_graph, self.root, self.ub_length)

        y = self.variables['y']
        x = self.model.addVars(flow_graph.edges, lb=0, vtype='C', name='x')
        self.variables['x'] = x

        # flow constraints
        for v in flow_graph.nodes:
            if v == self.root:
                continue

            if isinstance(flow_graph, nx.MultiDiGraph):
                sum_in = sum(x[a] for a in flow_graph.in_edges(v, keys=True))
                sum_out = sum(x[a] for a in flow_graph.out_edges(v, keys=True))
            else:
                sum_in = sum(x[a] for a in flow_graph.in_edges(v))
                sum_out = sum(x[a] for a in flow_graph.out_edges(v))

            self.model.addLConstr(sum_in - sum_out == y[v], f"flow_constr_{v}")

        if not self.params.add_z_variables:

            # arc has flow => head is active
            for a in flow_graph.edges:
                # couple x y
                self.model.addLConstr(x[a] <= y[a[1]] * big_M[a], f"coupl_x_y_{a}")

            if c2f:
                # do c2f coupling without z vars here.
                for a, fine_path in self.coarse_graph.proper_coarse_arcs:
                    # activate all fine nodes "under" coarse arc
                    for v_f in fine_path:
                        self.model.addLConstr(x[a] <= y[v_f] * big_M[a], f"activate_interior_{a}_{v_f}")

        else:

            # binary edge vars
            z = self.model.addVars(flow_graph.edges, vtype='B', name='z')
            # z = self.model.addVars(flow_graph.edges, lb=0, ub=1, vtype='C', name='z')
            self.variables['z'] = z

            # arc has flow => head is active
            for a in flow_graph.edges:

                # couple x z
                self.model.addLConstr(x[a] <= z[a] * big_M[a], f"coupl_x_z_{a}")

                # activate y, strengthened to 2-cycle cuts
                a_rev = a[::-1] if len(a) == 2 else (a[1], a[0], a[2])
                self.model.addLConstr(z[a] + z[a_rev] <= y[a[1]], f"coupl_z_{a}_y")

            # indegree
            for v in flow_graph.nodes:
                if v == self.root:
                    continue
                if isinstance(flow_graph, nx.MultiDiGraph):
                    sum_in = sum(z[a] for a in flow_graph.in_edges(v, keys=True))
                else:
                    sum_in = sum(z[a] for a in flow_graph.in_edges(v))
                self.model.addLConstr(sum_in == y[v], f'indegree_{v}')
        return

    ###########################################################
    # c2f scf

    def setup_c2f_scf_conn(self, allow_multiarcs=True):
        """
        We partition the vertices of a graph G = (V, E) into two sets: fine_nodes, the set of all vertices with degree
        <= 2, and coarse_nodes all with degree > 2. Then we construct a coarse_graph = (coarse_nodes, coarse_edges) from
        such that two coarse_nodes have an edge if and only if it exists a path in G containing just fine_nodes
        (except both endpoints), we call them fine_path. Note: Every edge in coarse_edges corresponds to a path in G.
        For the IP model we make use of coarse_graph such that we try to find a LR-MWCS in coarse_graph by using a
        single-commodity flow where for each chosen edge the vertices on the corresponding path are chosen as
        well. This is the coarse_flow. To select only subpaths of a fine_path we introduce the fine_flow.
        We construct a flow such that every coarse_node is a sink and as soon a vertex is chosen, there has to
        exists a flow to a sink.
        """
        coarse_nodes = set(v for v, d in self.core_graph.degree if d > 2).union([self.root])
        self.coarse_graph = util.build_coarse_graph(self.core_graph, coarse_nodes, allow_multiarcs=allow_multiarcs)

        # variables and constraints for coarse flow
        # self.setup_coarse_flow_ip()
        self.setup_scf_conn(flow_graph=self.coarse_graph, c2f=True)

        # fine flow ----------------------------------
        if self.params.fine_path_model == 'flow':
            self.add_fine_flow_with_big_m()
        else:
            self.add_fine_flow_with_pred()

        self.coarse_fine_coupling()

        return

    def add_fine_flow_with_big_m(self):

        fine_arcs, fine_nodes = util.get_fine_arcs(self.coarse_graph, mode='big_m')

        # variables for fine flow
        xf = self.model.addVars(fine_arcs, vtype='C', lb=0, name='xf')
        self.variables['xf'] = xf

        y = self.variables['y']

        coarse_nodes = set(self.coarse_graph.nodes)

        # flow constraints at fine nodes
        for v in fine_nodes:
            sum_in = sum(xf[u, v] for u in self.core_graph.neighbors(v))
            sum_out = sum(xf[v, w] for w in self.core_graph.neighbors(v) if w not in coarse_nodes)
            self.model.addLConstr(sum_in - sum_out == y[v], f"fine_flow_constr_{v}")

        for (u, v), big_m in fine_arcs.items():
            # the head of each arc must be active
            self.model.addLConstr(xf[u, v] <= y[v] * big_m, f'coupling_xf_y_{u}_{v}')

            if u in coarse_nodes:
                # if a fine arc starts at a coarse node, the coarse node must be active
                self.model.addLConstr(xf[u, v] <= y[u] * big_m, f'activate_coarse_{u}_{v}')

        return

    def add_fine_flow_with_pred(self):
        """
        For a fine arc (v, w) we store the predecessor u of v, to activate
        also the arc (u, v). If v is a coarse node, u is set to None.
        Attention: For the first fine arc of a fine path, there is only one direction
        (i.e., there is no variable for a fine arc with coarse head).
        """
        fine_arcs, fine_nodes = util.get_fine_arcs(self.coarse_graph, mode='pred')

        # variables for fine flow
        zf = self.model.addVars(fine_arcs, vtype='B', name='zf')
        self.variables['zf'] = zf

        y = self.variables['y']

        # fine node active => must be reached by arc
        for v in fine_nodes:
            sum_in = sum(zf[u, v] for u in self.core_graph.neighbors(v))
            self.model.addLConstr(sum_in == y[v], f"fine_arc_activation_{v}")

        for (v, w), u in fine_arcs.items():
            # fine arc is active => tail is active (stregthen with flow on other direction if possible)
            self.model.addLConstr(zf[v, w] + zf.get((w, v), 0) <= y[v], f'fine_tail_activation_{v}_{w}')

            # fine arc is active => previous fine arc is active
            if u is not None:
                self.model.addLConstr(zf[v, w] <= zf[u, v], f'fine_pred_arc_{v}_{w}')
        return

    def coarse_fine_coupling(self):

        if 'z' not in self.variables:
            # coupling was done in flow because we need big_M
            return

        y = self.variables['y']
        z = self.variables['z']

        a_done = set()
        for a, fine_path in self.coarse_graph.proper_coarse_arcs:
            a_rev = a[::-1] if len(a) == 2 else (a[1], a[0], a[2])
            if a_rev in a_done:
                continue
            a_done.add(a)
            # activate all fine nodes "under" coarse arc strengthened to quasi 2-cycle
            for v_f in fine_path:
                # self.model.addLConstr(z[a] <= y[v_f], f"activate_interior_{a}_{v_f}")
                self.model.addLConstr(z[a] + z[a_rev] <= y[v_f], f"activate_interior_{a}_{v_f}")

    ###########################################################
    # arc separator

    def setup_arc_sep_conn(self, digraph=None):
        """
        This is the SA_r model from
        "The Rooted Maximum Node-Weight Connected Subgraph Problem" by
        Eduardo Alvarez-Miranda, Ivana Ljubic, and Petra Mutzel (2013)

        The idea is that the x variables span an arborescence rooted at r.

        Connectivity is enforced via separation of the following constraints:

            z(S) >= y_v     forall v!=r, forall r-v-cuts S
        """
        if digraph is None:
            digraph = self.opt_graph
        else:
            self.opt_graph = digraph

        y = self.variables['y']

        # add binary edge variable -------------------------
        z = self.model.addVars(digraph.edges, vtype='B', name='z')
        # z = self.model.addVars(digraph.edges, lb=0, ub=1, vtype='C', name='z')
        self.variables['z'] = z

        # activate nodes of arborescence (if something goes in)
        for v in digraph.nodes:
            if v == self.root:
                continue
            self.model.addLConstr(sum(z[u, v] for u in digraph[v]) == y[v])

        # strengthening inequalities
        ############################
        for u, v in digraph.edges:
            # 2-cycle inequalities to strengthen formulation
            self.model.addLConstr(z[u, v] + z[v, u] <= y[v])

        self.prepare_arc_separation(digraph)

    def prepare_arc_separation(self, digraph):

        node_map = {v: i for i, v in enumerate(digraph.nodes)}
        self.model.Params.LazyConstraints = 1

        is_c2f_model = self.coarse_graph is not None

        self.model._callback_fun = separation.cb_arc_separator_cuts
        # give new attributes to model to use those in the callback
        self.model._myvars = self.variables
        self.model._nodes = list(digraph.nodes)
        self.model._node_map = node_map
        self.model._mapped_arcs = [(node_map[u], node_map[v]) for u, v in digraph.edges]
        self.model._num_nodes = digraph.number_of_nodes()
        self.model._digraph = digraph
        self.model._digraph2 = nx.convert_node_labels_to_integers(digraph)
        self.model._is_c2f = is_c2f_model
        self.model._backcuts = self.params.backcuts
        self.model._num_nested_cuts = self.params.num_nested_cuts
        # self.model._cut_threshold = 1e-6
        self.model._cut_threshold = 1 - 1e-3
        # self.model._cut_threshold = 0.5
        self.model._cb_cnt = 0
        self.model._cb_cut_cnt = 0
        self.model._root = self.root
        # self.model._full_graph = self.problem.graph  # for plotting

    #####################################################################
    # c2f arc separator

    def setup_c2f_arc_sep_conn(self):
        """
        arc separator model for the coarse graph
        """
        coarse_nodes = set(v for v, d in self.core_graph.degree if d > 2).union([self.root])
        self.coarse_graph = util.build_coarse_graph(self.core_graph, coarse_nodes, allow_multiarcs=False)

        self.setup_arc_sep_conn(digraph=self.coarse_graph)

        # fine flow ----------------------------------
        if self.params.fine_path_model == 'flow':
            self.add_fine_flow_with_big_m()
        else:
            self.add_fine_flow_with_pred()

        self.coarse_fine_coupling()

        return

    ###########################################################
    # Hybrid model

    def setup_hybrid_conn(self):

        self.setup_c2f_scf_conn(allow_multiarcs=False)

        self.prepare_arc_separation(self.opt_graph)

    ###########################################################
    # Node separator

    def setup_node_sep_conn(self):
        """
        This is the CUT_r model from
        "The Rooted Maximum Node-Weight Connected Subgraph Problem" by
        Eduardo Alvarez-Miranda, Ivana Ljubic, and Petra Mutzel (2013)

        The idea is the following: If a node v is active but not connected
        to the root, then some node in any v-r-separator has to be active as well.

        This is enforced via separation of the following constraints:

            y(N) >= y_v     forall v!=r, forall v-r-separators N
        """
        # TODO add cycle /neighbor constraints

        if self.params.add_z_variables:
            # add binary edge variable -------------------------
            digraph = self.opt_graph
            z = self.model.addVars(digraph.edges, vtype='B', name='z')
            self.variables['z'] = z

            y = self.variables['y']

            # activate nodes of arborescence (if something goes in) (indegree)
            for v in digraph.nodes:
                if v == self.root:
                    continue
                self.model.addLConstr(sum(z[u, v] for u in digraph[v]) == y[v])

            # strengthening inequalities
            ############################
            for u, v in digraph.edges:
                # 2-cycle inequalities to strengthen formulation
                self.model.addLConstr(z[u, v] + z[v, u] <= y[v])

        self.model.Params.LazyConstraints = 1

        # self.model._callback_fun = separation.cb_node_separator_cuts
        self.model._callback_fun = separation.cb_node_separator_integral
        self.model._myvars = self.variables
        self.model._graph = self.opt_graph
        self.model._backcuts = True
        self.model._cb_cnt = 0
        self.model._cb_cut_cnt = 0
        self.model._root = self.root

    ###########################################################
    # Multi-commodity flow

    def setup_mcf_conn(self):

        # TODO Dauert ziemlich lange :/

        logger.debug(f'Setting up mcf for root {self.root}')

        # only the following arcs are necessary for the flow
        flow_graph = self.opt_graph
        flow_nodes = [v for v in flow_graph.nodes if v != self.root]

        y = self.variables['y']

        # one flow variable per node
        x = {v: self.model.addVars(flow_graph.edges, lb=0, ub=1, vtype='C') for v in flow_nodes}
        self.variables['x'] = x

        # flow conservation
        for v in flow_nodes:
            for w in flow_nodes:
                sum_in = sum([x[v][u, w] for u in flow_graph.neighbors(w)])
                sum_out = sum([x[v][w, u] for u in flow_graph.neighbors(w)])
                if v != w:
                    self.model.addLConstr(sum_in - sum_out == 0, f"flow_cons_{v}_{w}")
                else:
                    self.model.addLConstr(sum_in == y[v])
                    self.model.addLConstr(sum_out == 0)

                for u in flow_graph.neighbors(v):
                    self.model.addLConstr(x[w][u, v] + x[w][v, u] <= y[v])

    ###########################################################
    # Heuristic

    def set_heuristic_warmstart(self, sol_ws=False):
        """
        If sol_ws, we give the solution as warmstart.
        Else, we give the objVal as lower bound.
        """
        heur_constrs = self.setup_tree_heuristic_conn()
        self.optimize()

        if self.model.status != gurobipy.GRB.Status.OPTIMAL:
            self.model.remove(heur_constrs)
            return

        if sol_ws:
            sol_nodes = self.get_solution()
            logger.info(f'Added heuristic solution with objective value {round(self.model.ObjVal, 3)} as warmstart')
            self.model.remove(heur_constrs)
            for v, vari in self.variables['y'].items():
                vari.start = v in sol_nodes
            return
        else:
            heur_obj_val = self.model.ObjVal
            logger.info(f'Added heuristic obj val {heur_obj_val} as lower bound.')
            self.model.addLConstr(
                sum(v * v.obj for v in self.variables['y'].values()) >= heur_obj_val, 'warm_start_lb'
            )
            self.model.remove(heur_constrs)

    def setup_tree_heuristic_conn(self):
        """
        Replace graph by BFS tree. Handle connectivity with implied nodes.

        :return:
        """
        bfs_pred = dict(nx.bfs_predecessors(self.graph, self.root))
        y = self.variables['y']

        heur_constrs = self.model.addConstrs((y[u] >= y[v] for v, u in bfs_pred.items()), name='pred')

        self.proper_core_graph = False
        return heur_constrs

    def change_graph(self, graph):
        """
        needed for sh_path_heuristic_2, where the graph is adjusted.
        We assume that the nodes are still the same.

        """
        self.graph = graph.to_directed()

        self.core_graph = graph

        self.opt_graph = self.graph

    ####################################################################################################################
    ####################################################################################################################

    def save_to_json(self, filepath):

        params = {'delta': self.delta,
                  'lb_length': self.lb_length,
                  'ub_length': self.ub_length,
                  'root': self.root}
        json_problem = {'params': params,
                        'graph': json_graph.node_link_data(self.graph)}

        with open(filepath, 'w') as f:
            json.dump(json_problem, f, indent=2)
        logger.info(f'\nProblem instance written to {filepath}\n')

    def debug_plot(self, nodes=None):
        if nodes is None:
            nodes = []
        g2 = nx.to_undirected(self.graph)
        # ksplot.draw_bwr(g2)
        node_colors = dict(self.graph.nodes(data='color'))
        draw.draw(g2, root=self.problem.root, node_color_dict=node_colors, colormap='bwr', highlight_node_list=nodes)
