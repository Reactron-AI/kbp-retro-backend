import argparse
import json
import os
import sys

from retro_star.case_study import case_study_result_path, collect_unique_depth0_routes
from retro_star.route_utils import canonical_smiles_set, json_safe

MODEL = 'interretro'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_mol', 
        default='OC[C@H]1O[C@@H](c2ccc(Cl)c(Cc3ccc(O[C@H]4CCOC4)cc3)c2)[C@H](O)[C@@H](O)[C@@H]1O'
        # default='[H][C@]12CCCCCCCCCCC(=O)N[C@@]11C(=O)C=C(CC)C(=O)[C@@]1([H])CC=C2'
    )
    parser.add_argument('--planner', choices=['mlp', 'localretro', 'interretro'], default=MODEL)
    parser.add_argument('--case_study_dir', default='case_study_' + MODEL)
    parser.add_argument('--unique_depth0_routes', action='store_true', default=True)
    parser.add_argument('--expansion_topk', type=int, default=50)
    parser.add_argument('--max_routes', type=int, default=100)
    parser.add_argument('--include_bb', nargs='*', default=[
        # Empagliflozin
        
        # Osimertinib
        # 'CN(C)C=CC(=O)c1cn(C)c2ccccc12', 'C=CC(=O)Cl', 'CNCCN(C)C', 'O=N[O-]', 'N#CN', 'COc1ccc(N)c(F)c1',
        # '[NH4+]', 'Cn1ccc2ccccc21', 'C1COCCN1', 'COc1cc(F)c(N)cc1N', 'CNCCN(C)C', 'C=CC(=O)Cl', 'O=CO', 'CC(=O)O',
        # 'Clc1ccnc(Cl)n1', 'Cn1ccc2ccccc21', 'O=C(Cl)CCCl', 'CNCCN(C)C', 'COc1cc(F)c([N+](=O)[O-])cc1N',
        # 'Clc1ccnc(Cl)n1', 'Cn1ccc2ccccc21', 'C=CC(=O)Cl', 'CNCCN(C)C', 'COc1cc(F)c([N+](=O)[O-])cc1N',
        # 'CNCCN(C)C', 'O=C(Cl)CCCl', 'COc1cc(F)c([N+](=O)[O-])cc1N', 'CC=O',
    ])
    parser.add_argument('--exclude_bb', nargs='*', default=[
        # Empagliflozin
        'OC[C@H]1O[C@@H](c2ccc(Cl)c(Cc3ccc(O[C@H]4CCOC4)cc3)c2)[C@H](O)[C@@H](O)[C@@H]1O',
        
        # Osimertinib
        # 'C=C',
        # 'CC',
        # 'C=O',
        # 'COC',
        # 'O=C=O',
        # 'CO',
        # 'O',
        # 'C',
        # 'COCOC',
        # 'CCOCOCC',
        # 'C1COCOC1',
        # 'CI',
        # 'CBr',
        # 'CCl',
        # 'BrBr',
        # 'II',
        # 'CC(=O)NBr',
        # 'O=C(O)CI',
        # 'BrC(Br)(Br)Br',
        # 'BrP(Br)Br',
        # 'C[Si](C)(C)CF',
        # 'C[Si](C)(C)CCl',
        # 'C[Si](C)(C)CBr',
        # 'C[Si](C)(C)CI',
        
        # 'CN1C=C(C2=NC(NC3=CC(NC(C=C)=O)=C(N(CCN(C)C)C)C=C3OC)=NC=C2)C4=CC=CC=C41',
        # 'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C',
        # 'C=CC(=O)Nc1cc(Nc2nccc(-c3c[nH]c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C',
        # 'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCNC',
        # 'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)C',
        # 'COc1cc(N(C)CCN(C)C)c(N)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
        # 'COc1cc(N(C)CCN(C)C)c([N+](=O)[O-])cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
        # 'COc1cc(F)c([N+](=O)[O-])cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
        # 'Cn1cc(-c2ccnc(Cl)n2)c2ccccc21',
        # 'Cn1cc(-c2ccnc(N)n2)c2ccccc21',
        # 'C=CC(=O)Nc1ccc(OC)c(N)c1',
        # 'COc1cc(N(C)CCN(C)C)c(N)cc1F',
        # 'O=C1CCC(=O)N1I',
        # 'O=C1CCC(=O)N1Cl',
        # 'O=C1CCC(=O)N1Br',
        # 'O=C1CCC(=O)N1F',
    ])
    parser.add_argument('--localretro_root', default=None)
    parser.add_argument('--localretro_model_path', default=None)
    parser.add_argument('--localretro_config_path', default=None)
    parser.add_argument('--localretro_data_dir', default=None)
    parser.add_argument('--interretro_root', default=None)
    parser.add_argument('--interretro_checkpoint', default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--starting_molecules', default='retro_star/dataset/origin_dict.csv')
    parser.add_argument('--no_value_fn', action='store_false', dest='use_value_fn')
    parser.add_argument('--no_viz', action='store_false', dest='viz')
    parser.add_argument('--viz_dir', default='viz')
    parser.set_defaults(use_value_fn=True, viz=True)
    return parser.parse_args()


def _canonical_smiles_set(smiles_list):
    return canonical_smiles_set(smiles_list)


def _build_planner(args):
    common_kwargs = {
        'gpu': args.gpu,
        'expansion_topk': args.expansion_topk,
        'root_expansion_topk': 'rank_after_banned' if args.unique_depth0_routes else None,
        'root_keep_first_reaction': args.unique_depth0_routes,
        'iterations': args.iterations,
        'starting_molecules': args.starting_molecules,
        'include_bb': args.include_bb,
        'exclude_bb': args.exclude_bb,
        'viz': args.viz,
        'viz_dir': args.viz_dir,
    }

    # retro_star.common parses sys.argv on import, so keep example-only args local.
    sys.argv = [sys.argv[0]]
    from retro_star.api import RSPlanner, RSPlannerInterRetro, RSPlannerLocalRetro

    if args.planner == 'localretro':
        return RSPlannerLocalRetro(
            **common_kwargs,
            use_value_fn=False,
            localretro_root=args.localretro_root,
            localretro_model_path=args.localretro_model_path,
            localretro_config_path=args.localretro_config_path,
            localretro_data_dir=args.localretro_data_dir
        )
    if args.planner == 'interretro':
        return RSPlannerInterRetro(
            **common_kwargs,
            use_value_fn=args.use_value_fn,
            interretro_root=args.interretro_root,
            interretro_checkpoint=args.interretro_checkpoint,
            localretro_root=args.localretro_root,
            localretro_model_path=args.localretro_model_path,
            localretro_config_path=args.localretro_config_path,
            localretro_data_dir=args.localretro_data_dir
        )
    return RSPlanner(
        **common_kwargs,
        use_value_fn=args.use_value_fn
    )


if __name__ == '__main__':
    args = parse_args()
    planner = _build_planner(args)

    if args.unique_depth0_routes:
        result = collect_unique_depth0_routes(
            planner,
            args.target_mol,
            args.case_study_dir,
            planner_name=args.planner,
            max_routes=args.max_routes,
        )
        result_path = case_study_result_path(args.case_study_dir)
        result['result_file'] = result_path
        os.makedirs(args.case_study_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(json_safe(result), f, indent=2)
    else:
        result = planner.plan(args.target_mol)

    print(json.dumps(json_safe(result), indent=2))
