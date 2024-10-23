import itertools
import logging

import networkx as nx
import networkx.algorithms.approximation.clique as nxca
import brcmwcs.draw as draw


logger = logging.getLogger()

class ConflictGraph:

    def __init__(self, problem, conflict_pairs):

        self.graph = problem.prep_graph.to_undirected()

        self.graph.remove_edges_from(problem.prep_graph.edges())

        self.graph.add_edges_from(conflict_pairs)

        # draw.draw_bwr(self.graph)

    def find_clique_covering(self, problem):
        if not self.graph.edges:
            return []
        if self.graph.number_of_edges() > 1000:
            logger.warning('Abort finding edge clique cover.')
            return list(self.graph.edges())

        # the following is only for logger purposes
        max_cliques = [c for c in nx.find_cliques(self.graph) if len(c) > 1]

        clique_size = {}
        for c in max_cliques:
            c_size = len(c)
            clique_size.setdefault(c_size, [])
            clique_size[c_size].append(c)

        logger.debug(f'Found an edge clique covering with {len(max_cliques)} cliques:')
        for s, c_list in clique_size.items():
            logger.debug(f'\t{len(c_list)} cliques of size {s}')

        return max_cliques

    def find_babo_cliques(self, problem):

        logger.debug('Compute shortest paths..')
        # TODO if weight schon an Kanten, dann nicht hinschreiben
        weight = {(u, v): self.graph.nodes[v]['length'] for u, v in problem.prep_graph.edges}
        nx.set_edge_attributes(problem.prep_graph, weight, 'weight')

        spl = nx.single_source_dijkstra_path_length(problem.prep_graph, problem.root)
        # nodes50 = [v for v in spl if spl[v] <= 50]
        # cg2 = self.graph.subgraph(nodes50)
        # draw.draw_node_weighted_graph2(cg2, root=problem.root, color_map='Set1',
        #                                traffic_map={v: 1 for v in problem.prep_graph.nodes})
        # draw.draw_bwr(cg2,root=problem.root)
        cliques = []

        find_all_cliques = False
        find_all_cliques = True

        logger.debug('Find cliques...')
        clique_added = True
        while clique_added:

            clique_added = False

            if find_all_cliques:

                # Returns all maximal cliques in an undirected graph.
                cliques = list(nx.find_cliques(self.graph))
                max_cliques = dict(enumerate(set(c) for c in cliques))

                cliques.sort(key=lambda c: len(c), reverse=True)
                solution_nodes = [122, 123, 125, 139, 141, 207, 212, 213, 234, 236, 247, 248, 250, 251, 265, 289, 290, 291, 294, 329, 331, 416, 511, 513, 531, 543, 544]
                draw.draw_node_weighted_graph2(problem.prep_graph.to_undirected(), root=problem.root,
                                               highlight_node_list=cliques[100], color_map='Set1',
                                               traffic_map={v: 1 for v in problem.prep_graph.nodes},
                                               solution_nodes=solution_nodes)

                for clique in cliques:
                    draw.draw_node_weighted_graph2(problem.prep_graph.to_undirected(), root=problem.root,
                                                   highlight_node_list=clique, color_map='Set1',
                                                   traffic_map={v: 1 for v in problem.prep_graph.nodes})

                mc_graph = nx.Graph()
                mc_graph.add_nodes_from(max_cliques.keys())

                clique_pairs = itertools.combinations(max_cliques.items(), 2)
                mc_graph.add_edges_from((i, j) for (i, c1), (j, c2) in clique_pairs if c1 & c2)

                while max_cliques:

                    logger.info('here')

                    # find the largest clique with minimum sum of lengths to root
                    best_cliques = []
                    for i in range(2, 6):
                        rel_cliques = [p[1] for p in max_cliques.items() if len(p[1]) == i]
                        if rel_cliques:
                            best_cliques.append(max(rel_cliques, key=lambda c: (len(c), -sum(spl[v] for v in c))))

                    for c in best_cliques:
                        draw.draw_node_weighted_graph2(problem.prep_graph, root=problem.root,
                                                       highlight_node_list=c, color_map='Set1',
                                                       traffic_map={v: 1 for v in problem.prep_graph.nodes})

                    i, best_clique = max(max_cliques.items(), key=lambda p: (len(p[1]), -sum(spl[v] for v in p[1])))

                    if len(best_clique) == 2:
                        break

                    clique_added = True
                    cliques.append(best_clique)
                    logger.debug(f'Best clique of size {len(best_clique)}.')

                    # remove edges of clique
                    self.graph.remove_edges_from(itertools.combinations(best_clique, 2))

                    # remove cliques that share a vertex with best_clique from max_cliques
                    remove_nodes = [i] + list(mc_graph.neighbors(i))
                    for j in remove_nodes:
                        del max_cliques[j]
                    mc_graph.remove_nodes_from(remove_nodes)

                    draw.draw_node_weighted_graph2(problem.prep_graph, root=problem.root,
                                                   highlight_node_list=best_clique, color_map='Set1',
                                                   traffic_map={v:1 for v in problem.prep_graph.nodes})

            else:
                # heuristic for best
                best_clique = nxca.max_clique(self.graph)

                if len(best_clique) == 2:
                    return cliques

                clique_added = True
                cliques.append(best_clique)
                logger.debug(f'Best clique of size {len(best_clique)}.')

                # remove edges of clique
                self.graph.remove_edges_from(itertools.combinations(best_clique, 2))

                draw.draw_node_weighted_graph2(problem.prep_graph, root=problem.root,
                                               highlight_node_list=best_clique, color_map='Set1',
                                               traffic_map={v: 1 for v in problem.prep_graph.nodes})
        return cliques

