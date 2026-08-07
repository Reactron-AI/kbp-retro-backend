import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from retro_star.common.prepare_utils import prepare_interretro, \
    prepare_molstar_planner, prepare_starting_molecules
from retro_star.route_utils import depth0_reaction_signature, route_steps, \
    route_tree_segments, signature_to_report, write_route_artifacts
from retro_star.utils import setup_logger


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--test_routes',
                        default='dataset/routes_possible_test_hard.pkl')
    parser.add_argument('--starting_molecules', default='dataset/origin_dict.csv')

    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--expansion_topk', type=int, default=50)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--viz_dir', default='viz')

    parser.add_argument('--use_value_fn', action='store_true')
    parser.add_argument('--result_folder', default='results')
    parser.add_argument('--unique_depth0_routes', action='store_true')
    parser.add_argument('--max_routes', type=int, default=5)
    parser.add_argument('--case_study_dir', default='case_study')

    parser.add_argument('--interretro_root', default=None)
    parser.add_argument('--interretro_checkpoint', default=None)
    parser.add_argument('--localretro_root', default=None)
    parser.add_argument('--localretro_model_path', default=None)
    parser.add_argument('--localretro_config_path', default=None)
    parser.add_argument('--localretro_data_dir', default=None)

    return parser


def _case_study_route_path(case_study_dir, route_idx):
    return os.path.join(case_study_dir, 'routes', 'mol_%s_route.txt' % route_idx)


def _case_study_viz_path(case_study_dir, route_idx):
    return os.path.join(case_study_dir, 'viz', 'mol_%s_route.png' % route_idx)


def _collect_unique_depth0_routes(target_mol, target_mol_id, plan_handle, args):
    routes_dir = os.path.join(args.case_study_dir, 'routes')
    viz_dir = os.path.join(args.case_study_dir, 'viz')
    os.makedirs(routes_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    banned_reactions = set()
    depth0_signatures = []
    routes = []
    skipped_duplicate_root_routes = 0
    stop_reason = 'max_routes'
    t0 = time.time()

    search_idx = 0
    while len(routes) < args.max_routes:
        current_search_idx = search_idx
        search_idx += 1
        succ, msg = plan_handle(
            target_mol,
            current_search_idx,
            viz_override=False,
            viz_dir_override=None,
            banned_reactions=banned_reactions,
            return_mol_tree=True
        )
        if not succ:
            stop_reason = 'no_route_found'
            break

        route, iteration_count, _ = msg
        steps = route_steps(route)
        depth0_signature = depth0_reaction_signature(steps)
        if not depth0_signature:
            stop_reason = 'no_depth0_reaction'
            break

        banned_reactions.update(depth0_signature)
        if depth0_signature in depth0_signatures:
            skipped_duplicate_root_routes += 1
            continue

        route_idx = len(routes)
        depth0_signatures.append(depth0_signature)
        artifact_idx = current_search_idx
        if target_mol_id is not None:
            artifact_idx = '%d_search_%d' % (target_mol_id, current_search_idx)
        route_file = _case_study_route_path(args.case_study_dir, artifact_idx)
        route_viz = _case_study_viz_path(args.case_study_dir, artifact_idx)
        viz_written = write_route_artifacts(route, route_file, route_viz)

        routes.append({
            'route_index': route_idx,
            'search_index': current_search_idx,
            'artifact_index': artifact_idx,
            'succ': True,
            'iter': iteration_count,
            'route_file': route_file,
            'route_viz': route_viz if viz_written else None,
            'route': route.serialize(),
            'route_tree_segments': route_tree_segments(steps),
            'steps': steps,
            'depth0_reaction_signature': signature_to_report(depth0_signature),
            'route_cost': float(route.total_cost),
            'route_len': route.length,
            'optimal': route.optimal,
        })

    return {
        'target_mol': target_mol,
        'target_mol_id': target_mol_id,
        'novelty_policy': 'depth0_unique_reactions',
        'max_routes': args.max_routes,
        'search_count': search_idx,
        'elapsed_time': time.time() - t0,
        'routes': routes,
        'route_signatures': [
            signature_to_report(signature)
            for signature in depth0_signatures
        ],
        'depth0_reaction_signatures': [
            signature_to_report(signature)
            for signature in depth0_signatures
        ],
        'skipped_duplicate_root_routes': skipped_duplicate_root_routes,
        'stop_reason': stop_reason,
    }


def retro_plan(args):
    device = torch.device('cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu')

    starting_mols = prepare_starting_molecules(args.starting_molecules)

    with open(args.test_routes, 'rb') as f:
        routes = pickle.load(f)
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.test_routes))

    one_step = prepare_interretro(
        interretro_root=args.interretro_root,
        checkpoint_path=args.interretro_checkpoint,
        localretro_root=args.localretro_root,
        localretro_model_path=args.localretro_model_path,
        localretro_config_path=args.localretro_config_path,
        localretro_data_dir=args.localretro_data_dir,
        topk=args.expansion_topk,
        device=device
    )

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    value_fn = one_step.value if args.use_value_fn else lambda x: 0.

    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )

    if args.unique_depth0_routes:
        os.makedirs(args.case_study_dir, exist_ok=True)
        case_study = {
            'planner': 'interretro',
            'test_routes': args.test_routes,
            'novelty_policy': 'depth0_unique_reactions',
            'max_routes': args.max_routes,
            'targets': []
        }
        for i, route in enumerate(routes):
            target_mol = route[0].split('>')[0]
            case_study['targets'].append(
                _collect_unique_depth0_routes(target_mol, i, plan_handle, args)
            )

        return case_study

    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': []
    }
    num_targets = len(routes)
    t0 = time.time()
    for i, route in enumerate(routes):
        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(target_mol, i)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))

    with open(args.result_folder + '/plan_interretro.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan_interretro.log')

    retro_plan(args)
