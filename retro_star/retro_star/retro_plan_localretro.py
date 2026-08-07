import argparse
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from retro_star.common import prepare_localretro, prepare_molstar_planner, \
    prepare_starting_molecules, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.route_utils import json_safe, route_steps, route_tree_segments, \
    write_route_artifacts
from retro_star.utils import setup_logger


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--test_routes',
                        default='dataset/routes_possible_test_hard.pkl')
                        # default='dataset/target_molecule_list.pkl')
    parser.add_argument('--starting_molecules', default='dataset/origin_dict.csv')

    parser.add_argument('--value_root', default='dataset')
    parser.add_argument('--value_train', default='train_mol_fp_value_step')
    parser.add_argument('--value_val', default='val_mol_fp_value_step')

    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--expansion_topk', type=int, default=50)
    parser.add_argument('--viz', default=True)
    parser.add_argument('--viz_dir', default='viz')
    parser.add_argument('--case_study_dir', default='case_study_localretro')

    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)

    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_epoch_int', type=int, default=1)
    parser.add_argument('--save_folder', default='saved_models')

    parser.add_argument('--use_value_fn', action='store_true')
    parser.add_argument('--value_model', default='best_epoch_final_4.pt')
    parser.add_argument('--result_folder', default='results')

    parser.add_argument('--localretro_root', default=None)
    parser.add_argument('--localretro_model_path', default=None)
    parser.add_argument('--localretro_config_path', default=None)
    parser.add_argument('--localretro_data_dir', default=None)

    return parser


def _case_study_route_path(case_study_dir, route_idx):
    return os.path.join(case_study_dir, 'routes', 'mol_%s_route.txt' % route_idx)


def _case_study_viz_path(case_study_dir, route_idx):
    return os.path.join(case_study_dir, 'viz', 'mol_%s_route.png' % route_idx)


def retro_plan(args):
    device = torch.device('cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu')

    starting_mols = prepare_starting_molecules(args.starting_molecules)

    with open(args.test_routes, 'rb') as f:
        routes = pickle.load(f)
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.test_routes))

    one_step = prepare_localretro(
        localretro_root=args.localretro_root,
        model_path=args.localretro_model_path,
        config_path=args.localretro_config_path,
        data_dir=args.localretro_data_dir,
        device=device
    )

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.use_value_fn:
        model = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1, -1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )

    result = {
        'target_mols': [],
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': [],
        'route_files': [],
        'route_viz': []
    }
    case_study = None
    if args.viz:
        os.makedirs(args.case_study_dir, exist_ok=True)
        case_study = {
            'planner': 'localretro',
            'test_routes': args.test_routes,
            'targets': []
        }
    num_targets = len(routes)
    t0 = time.time()
    for i, route in enumerate(routes):
        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(
            target_mol,
            i,
            viz_override=False if args.viz else None,
            viz_dir_override=None
        )

        result['target_mols'].append(target_mol)
        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        route_file = None
        route_viz = None

        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
            if args.viz:
                route_file = _case_study_route_path(args.case_study_dir, i)
                route_viz_path = _case_study_viz_path(args.case_study_dir, i)
                viz_written = write_route_artifacts(
                    msg[0],
                    route_file,
                    route_viz_path
                )
                route_viz = route_viz_path if viz_written else None
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        result['route_files'].append(route_file)
        result['route_viz'].append(route_viz)

        if case_study is not None:
            steps = route_steps(msg[0]) if succ else []
            case_study['targets'].append({
                'target_mol': target_mol,
                'target_mol_id': i,
                'succ': succ,
                'iter': msg[1],
                'route_file': route_file,
                'route_viz': route_viz,
                'route': msg[0].serialize() if succ else None,
                'route_tree_segments': route_tree_segments(steps),
                'steps': steps,
                'route_cost': float(msg[0].total_cost) if succ else None,
                'route_len': msg[0].length if succ else None,
                'optimal': msg[0].optimal if succ else None,
            })

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))

    with open(args.result_folder + '/plan_uspto50k.pkl', 'wb') as f:
        pickle.dump(result, f)

    if case_study is not None:
        with open(os.path.join(args.case_study_dir, 'routes.json'), 'w') as f:
            json.dump(json_safe(case_study), f, indent=2)


if __name__ == '__main__':
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan_uspto50k.log')

    retro_plan(args)
