import argparse
import datetime
import os
import sys
import logging
import time

code_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_repo)

# gurobipath = '/software/opt-sw/gurobi1003/linux64/lib/python3.11_utf32'
# if gurobipath not in sys.path:
#     sys.path.append(gurobipath)

# os.environ['GUROBI_HOME'] = '/software/opt-sw/gurobi1003/linux64'
# os.environ['LD_LIBRARY_PATH'] = '/software/opt-sw/gurobi1003/linux64/lib'
# os.environ['PATH'] = os.environ['PATH'] + '/software/opt-sw/gurobi1003/linux/bin'
# os.environ['GRB_LICENSE_FILE'] = '/software/opt-sw/gurobi/gurobi.lic'


import brcmwcs.draw as draw
import brcmwcs.Solution as sol
import brcmwcs.optimization as opt
import brcmwcs.Problem as prob

# import brcmwcs.stats as stats

logger = logging.getLogger()


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--problem_path',
                        help='Pfad zu einer Problem Instanz, die geladen werden soll.')
    parser.add_argument('--solution_path', type=str, default=None,
                        help='Pfad zu einer Solution Instanz, die geladen werden soll.')
    # parser.add_argument('--solution_path_good_cuts', type=str, default=None,
    #                     help='Pfad zu einer Solution Instanz, die geladen werden soll um good cuts zu extrahieren.')
    # parser.add_argument('--n_good_cuts', type=int, default=0,
    #                     help='Wie viele der besten cuts soll genommen werden?')
    parser.add_argument('--save_solution_to', type=str, default=None,
                        help='Falls gegeben, wird eine json für die Lösung dort gespeichert.')
    parser.add_argument('--save_log_to', type=str,
                        help='Falls gegeben, wird das logfile gespeichert.')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'ERROR'], default='INFO',
                        help='Genauigkeit der log Ausgabe')

    parser.add_argument('--lb', dest='lb_length', type=float,
                        help='lower bound (to overwrite).')
    parser.add_argument('--ub', dest='ub_length', type=float,
                        help='upper bound (to overwrite).')

    parser.add_argument('--connectivity_mode', type=str, default='arc_sep',
                        choices=['scf', 'c2f_scf', 'arc_sep', 'c2f_arc_sep', 'hybrid', 'node_sep', 'mcf',
                                 'tree_heuristic'],
                        help='Which connectivity model?')
    parser.add_argument('--use_warmstart_heur', type=int, default=0,
                        help='Use tree heuristic on BFS to potentially generate a warmstart?')
    parser.add_argument('--preprocessing', type=int, default=2, choices=[0, 1, 2],
                        help='run median preprocessing?\n'
                             '0 - no\n'
                             '1 - single color preprocessing (TODO?)\n'
                             '2 - bicolor radius')
    parser.add_argument('--fixed_nodes_preprocessing', type=int, default=1,
                        help='run fixed nodes preprocessing?')
    parser.add_argument('--find_implied_nodes', type=int, default=1,
                        help='compute and use implied nodes?')
    parser.add_argument('--filter_implied_nodes', type=int, default=0,
                        help='filter implied nodes?')
    parser.add_argument('--find_conflict_pairs_mode', type=int, default=0, choices=[0, 1, 2, 3],
                        help='find conflict pairs with Steiner trees?\n'
                             '0 - no\n'
                             '1 - only with color ranges\n'
                             '2 - with color ranges and length\n'
                             '3 - bicolor stein tree')
    parser.add_argument('--filter_conflicts', type=int, default=2, choices=[i for i in range(9)],
                        help='method for filtering conflict pairs\n'
                             '0 - no filter\n'
                             '1 - paper scoring function\n'
                             '2 - scoring function only on non dominated conflict pairs')
    parser.add_argument('--babo_cliques', type=int, default=0,
                        help='run optimization with babo cliques?')
    parser.add_argument('--knapsack_constraints', type=int, default=0, choices=[0, 1],
                        help='Add knapsack constraints to MIP?')
    parser.add_argument('--add_z_variables', type=int, default=1, choices=[0, 1],
                        help='Add binary z variables for formulations that do not use them (SCF, RNS)?')
    parser.add_argument('--root_ring_cuts', type=int, default=1, choices=[0, 1],
                        help='Add cuts that forces vertices in the near neighborhood of the root to be chosen?')
    parser.add_argument('--fine_path_cuts', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='Add cuts that forces finepath in the neighborhood of high profit nodes to be chosen?')
    parser.add_argument('--arc_neighborhood_cuts', type=int, default=2, choices=[0, 1, 2, 3],
                        help='Radius für extended indegree constraints')
    parser.add_argument('--arc_coarse_neighborhood_cuts', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Radius für extended indegree constraints on coarse nodes only')
    parser.add_argument('--add_indegree_constraints', type=int, default=0, choices=[0, 1],
                        help='Add cuts that force an ingoing arc to be choosen if an outgoing arc is chosen?')
    parser.add_argument('--branching_prio', type=int, default=0, choices=[0, 1],
                        help='Should we set branching priorities?')
    parser.add_argument('--fine_path_model', type=str, default="pred", choices=["flow", "pred"],
                        help='Model fine paths as flow or with predecessors?')
    parser.add_argument('--add_color_range_constr', type=int, default=0,
                        help='Add color range constraints derived from preprocessing?')
    parser.add_argument('--add_balance_constr', type=int, default=1,
                        help='Enforce that root is median in IP?')
    parser.add_argument('--backcuts', type=int, default=1,
                        help='Use backcuts in RAS/RNS separation?')
    parser.add_argument('--num_nested_cuts', type=int, default=1,
                        help='Use nested cuts in RAS separation? How many?')
    parser.add_argument('--solver', type=str, default='gurobi', choices=['gurobi'],
                        help='run optimization with solver gurobi?')
    parser.add_argument('--save_conflict_pairs_to', type=str,
                        help='Sollen alle conflict pairs gespeichert werden?')
    parser.add_argument('--save_prep_out_to', type=str,
                        help='Soll der Output vom Preprocessing gespeichert werden?')
    parser.add_argument('--load_prep_out_from', type=str,
                        help='Soll das Ergebnis des Preprocessing geladen werden?')
    parser.add_argument('--objective_cut', type=float, default=None,
                        help='Add a cut to say the optimal solution is better or equal.')

    params = parser.parse_args(args)

    logger.handlers = []
    if params.log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG,
                            format="%(filename)s, line %(lineno)s, %(levelname)s: %(message)s")
    elif params.log_level == "INFO":
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    elif params.log_level == "ERROR":
        logging.basicConfig(level=logging.ERROR, format="%(message)s")

    if params.save_log_to is not None:
        fh = logging.FileHandler(params.save_log_to, mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    if params.save_log_to is not None:
        fh = logging.FileHandler(params.save_log_to, mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    return params


def print_params(params):
    logger.info('\nbrcmwcs is run with the following parameters:')
    tab_len = 30
    for k, v in vars(params).items():
        line = '\t {0:{1}}:  {2}'.format(k, tab_len, v)
        logger.info(line)
    logger.info('')


def main(args):

    # params and start
    params = parse_arguments(args)
    logger.info(f'\nbrcmwcs started at {datetime.datetime.now().strftime("%Y-%b-%d, %H:%M:%S")}\n{"#"*40}')

    # make original problem
    problem, load_success = prob.load_problem_from_json(params)
    if not load_success:
        logger.error(f'Probleminstanz {params.problem_path} nicht gefunden.')
        sys.exit()
    print_params(problem.params)

    # try to load solution
    solution, load_success = sol.load_solution(problem.params)
    # draw.draw_bwr(problem.graph, problem.root)

    if not load_success:
        # preprocessing
        ###############
        prep_out = problem.run_preprocessing()
        solution.stats['prep_stats'] = {key: list(val) if type(val) == set else val for key, val in
                                        dict(vars(prep_out)).items()}
        if params.save_prep_out_to:
            prep_out.save_to_json(params.save_prep_out_to)
            return
        if not prep_out.feasible:
            logger.info('preprocessing proved infeasible')  # root impossible
            solution.stats['opt_time'] = 0
            return solution

        # optimization
        ##############
        print_gurobi = params.log_level != 'ERROR'
        opt_model = opt.OptModel(problem, print_gurobi=print_gurobi)

        opt_model.setup(set_objective=True)

        # opt_model.model.write('/nfs/homeoptimi/bzfschws/temp/brb_scf.lp')

        opt_model.optimize()

        solution = opt_model.get_solution(solution)

        # opt.optimize(problem, prep_out, solution)

        # logger.info(f'Preprocessing time: {prep_out.time}\t({len(prep_out.forbidden_pairs)} forbidden pairs)\n'
        #             f'Optimization time: {solution.stats["opt_time"]}')

        if params.save_solution_to is not None:
            solution.save_to_json(params.save_solution_to)

    # draw.draw(problem.graph, root=problem.root, node_list=solution.subgraph_nodes)

    return solution


if __name__ == '__main__':
    main(sys.argv[1:])

