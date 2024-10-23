

import argparse
import logging
import json
import os
import git
import networkx as nx

logger = logging.getLogger()


class Solution:

    def __init__(self, params):
        self.params = params
        self.subgraph_nodes = []
        self.stats = {}

    def check_solution(self, problem):
        # balance
        red_nodes = [v for v in problem.graph if problem.graph.nodes[v]['color'] == -1]
        red_weight = sum([problem.graph.nodes[v]['length'] for v in self.subgraph_nodes if v in red_nodes])

        blue_nodes = [v for v in problem.graph if problem.graph.nodes[v]['color'] == 1]
        blue_weight = sum([problem.graph.nodes[v]['length'] for v in self.subgraph_nodes if v in blue_nodes])

        logger.debug(f'rotes Gewicht: {red_weight}\nblaues Gewicht: {blue_weight}\nDelta: {problem.params.delta}')
        balanced = bool(abs(red_weight - blue_weight) <= problem.params.delta)
        if not balanced:
            logger.error('Der Graph ist nicht balanciert.')
            return False
        else:
            logger.debug(f'Der Graph ist balanciert.')

        # weight
        weight_feasible = bool(problem.params.lb_length <= red_weight + blue_weight <= problem.params.ub_length)
        logger.debug(f'Längenintervall: [{problem.params.lb_length},{problem.params.ub_length}]\n'
                     f'Gesamtgewicht: {red_weight + blue_weight}')
        if not weight_feasible:
            logger.error(f'Der Graph erfüllt nicht die Längenbeschränkung ({red_weight + blue_weight})')
            return False
        else:
            logger.debug('Der Graph erfüllt die Längenbeschränkung.')

        # connectivity
        subgraph = nx.subgraph(problem.graph, self.subgraph_nodes)
        if not nx.is_connected(subgraph):
            logger.error('Der Graph ist nicht zusammenhängend.')
            return False
        else:
            logger.debug('Der Graph ist zusammenhängend.')

        return True

    def save_to_json(self, filepath):
        save_stats = {k: v for k, v in self.stats.items() if k != 'prep_stats'}
        save_stats['prep_stats'] = {}

        if self.stats["prep_stats"]["stats"] is not None:
            for stat, val in vars(self.stats["prep_stats"]["stats"]).items():
                save_stats['prep_stats'][stat] = val
        #
        #
        # for v in self.stats["prep_stats"]["implied_nodes"]:
        #     self.stats["prep_stats"]["implied_nodes"][v] = list(self.stats["prep_stats"]["implied_nodes"][v])
        #
        # for stat, val in vars(self.stats["prep_stats"]["stats"]).items():
        #     self.stats["prep_stats"][stat] = val
        #
        # del (self.stats["prep_stats"]["stats"])

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        branch = repo.active_branch

        json_solution = {
            'params': vars(self.params),
            'stats': save_stats,
            'solution': self.subgraph_nodes,
            'git': {
                'branch_name': branch.name,
                'commit_hash': sha
            }
        }
        with open(filepath, 'w') as f:
            json.dump(json_solution, f, indent=2)
        logger.info('\nSolution instance written to {}\n'.format(filepath))


def load_solution(params):

    solution = Solution(params)

    if params.solution_path is None or not os.path.exists(params.solution_path):
        return solution, False
    with open(params.solution_path, 'r') as f:
        json_solution = json.load(f)

    sol_params = argparse.Namespace(**json_solution['params'])

    # TODO comparison of objects probably not correct.. <- Solution: Only check relevant parameters
    if (
            sol_params.root != params.root or
            sol_params.lb_length != params.lb_length or
            sol_params.ub_length != params.ub_length or
            sol_params.delta != params.delta or
            sol_params.problem_path != params.problem_path
    ):
        # solution does not fit to parameters
        return solution, False

    solution.stats = json_solution['stats']
    solution.subgraph_nodes = json_solution['solution']
    return solution, True


def load_solution_from_json(sol_path):

    if not os.path.exists(sol_path):
        return None, False

    with open(sol_path, 'r') as f:
        json_solution = json.load(f)

    sol_params = argparse.Namespace(**json_solution['params'])

    solution = Solution(sol_params)

    solution.stats = json_solution['stats']
    solution.subgraph_nodes = json_solution['solution']
    return solution, True
