import argparse
import logging
import json
import os

from networkx.readwrite import json_graph

import brcmwcs.preprocessing as prep

logger = logging.getLogger()


class Problem:

    def __init__(self,
                 graph,
                 params
                 ):

        self.graph = graph  # nx.Graph
        self.params = params

        self.root = params.root
        self.delta = params.delta
        self.lb_length = params.lb_length
        self.ub_length = params.ub_length

        self.prep_out = prep.PrepOutput()

    def save_to_json(self, filepath):
        json_problem = {'params': vars(self.params),
                        'graph': json_graph.node_link_data(self.graph)}

        with open(filepath, 'w') as f:
            json.dump(json_problem, f, indent=2)
        logger.info(f'\nProblem instance written to {filepath}\n')

    def run_preprocessing(self):

        if self.params.load_prep_out_from:
            # try to load prep_out
            self.prep_out.load_json(self.params.load_prep_out_from)
            if self.params.filter_conflicts:
                preprocessing = prep.Preprocessing(self, prep_out=self.prep_out)
                preprocessing.filter_conflict_pairs()
                self.prep_out.forbidden_pairs = preprocessing.forbidden_pairs

        elif self.params.preprocessing:
            preprocessing = prep.Preprocessing(self)
            self.prep_out = preprocessing.run_preprocessing()

        return self.prep_out


def load_problem_from_json(params):
    """
    Load the graph and parameters from the instance file.
    """
    filepath = params.problem_path
    if filepath is None:
        return None, False
    if not os.path.isfile(filepath):
        logger.error(f'>> ERROR: The instance file "{filepath}" could not be found.\n')
        return None, False
    if not filepath.endswith('.json'):
        logger.error(f'>> ERROR: The instance file "{filepath}" is not a json file.\n')
        return None, False

    logger.info(f'\nLoading Problem instance from file {filepath}...')
    with open(filepath, 'r') as f:
        json_problem = json.load(f)

    graph = json_graph.node_link_graph(json_problem['graph'])
    loaded_params = argparse.Namespace(**json_problem['params'])

    # overwrite attributes
    for att, val in vars(params).items():
        setattr(loaded_params, att, val)

    if params.lb_length is None:
        loaded_params.lb_length = json_problem['params']['lb_length']
    if params.ub_length is None:
        loaded_params.ub_length = json_problem['params']['ub_length']

    problem = Problem(params=loaded_params, graph=graph.to_undirected())

    logger.info('Done.')
    return problem, True
