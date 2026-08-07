import os
import time
import logging

import torch

from retro_star.common.prepare_utils import prepare_interretro, prepare_localretro, \
    prepare_mlp, prepare_molstar_planner, prepare_starting_molecules
from retro_star.common.smiles_to_fp import smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger


dirpath = os.path.dirname(os.path.abspath(__file__))


def _prepare_value_fn(device, use_value_fn, save_folder, value_model, fp_dim):
    if use_value_fn:
        model = ValueMLP(
            n_layers=1,
            fp_dim=fp_dim,
            latent_dim=128,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (save_folder, value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    return value_fn


def _as_building_block_set(building_blocks):
    if building_blocks is None:
        return None
    if isinstance(building_blocks, str):
        return {building_blocks}
    return set(building_blocks)


def _apply_building_block_filters(starting_mols, include_bb=None, exclude_bb=None):
    include_bb = _as_building_block_set(include_bb)
    exclude_bb = _as_building_block_set(exclude_bb)

    if exclude_bb:
        starting_mols = starting_mols.difference(exclude_bb)
        logging.info('%d starting molecules kept after exclude_bb filtering' %
                     len(starting_mols))

    if include_bb:
        starting_mols = starting_mols.union(include_bb)
        logging.info('%d starting molecules kept after include_bb union' %
                     len(starting_mols))

    return starting_mols


def _format_plan_result(succ, msg, elapsed, target_mol):
    if succ:
        return {
            'succ': succ,
            'time': elapsed,
            'iter': msg[1],
            'routes': msg[0].serialize(),
            'route_cost': msg[0].total_cost,
            'route_len': msg[0].length
        }

    logging.info('Synthesis path for %s not found. Please try increasing '
                 'the number of iterations.' % target_mol)
    return None


class RSPlanner:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 include_bb=None,
                 exclude_bb=None,
                 mlp_templates=dirpath+'/one_step_model/template_rules_1.dat',
                 mlp_model_dump=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt',
                 save_folder=dirpath+'/saved_models',
                 value_model='best_epoch_final_4.pt',
                 fp_dim=2048,
                 viz=True,
                 viz_dir='viz',
                 root_expansion_topk=None,
                 root_keep_first_reaction=False):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)
        starting_mols = _apply_building_block_filters(
            starting_mols=starting_mols,
            include_bb=include_bb,
            exclude_bb=exclude_bb
        )

        one_step = prepare_mlp(mlp_templates, mlp_model_dump)
        value_fn = _prepare_value_fn(
            device=device,
            use_value_fn=use_value_fn,
            save_folder=save_folder,
            value_model=value_model,
            fp_dim=fp_dim
        )

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            root_expansion_topk=root_expansion_topk,
            root_keep_first_reaction=root_keep_first_reaction
        )

    def plan_raw(self, target_mol, target_mol_id=0, viz=None, viz_dir=None,
                 banned_reactions=None, return_mol_tree=False,
                 root_rank_start=None, exclude_smiles=None):
        return self.plan_handle(
            target_mol,
            target_mol_id,
            viz_override=viz,
            viz_dir_override=viz_dir,
            banned_reactions=banned_reactions,
            return_mol_tree=return_mol_tree,
            root_rank_start=root_rank_start,
            exclude_smiles=exclude_smiles
        )

    def plan(self, target_mol):
        t0 = time.time()
        succ, msg = self.plan_raw(target_mol)
        return _format_plan_result(succ, msg, time.time() - t0, target_mol)


class RSPlannerLocalRetro:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 include_bb=None,
                 exclude_bb=None,
                 localretro_root=None,
                 localretro_model_path=None,
                 localretro_config_path=None,
                 localretro_data_dir=None,
                 save_folder=dirpath+'/saved_models',
                 value_model='best_epoch_final_4.pt',
                 fp_dim=2048,
                 viz=True,
                 viz_dir='viz',
                 root_expansion_topk=None,
                 root_keep_first_reaction=False):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)
        starting_mols = _apply_building_block_filters(
            starting_mols=starting_mols,
            include_bb=include_bb,
            exclude_bb=exclude_bb,
        )

        one_step = prepare_localretro(
            localretro_root=localretro_root,
            model_path=localretro_model_path,
            config_path=localretro_config_path,
            data_dir=localretro_data_dir,
            device=device
        )
        value_fn = _prepare_value_fn(
            device=device,
            use_value_fn=use_value_fn,
            save_folder=save_folder,
            value_model=value_model,
            fp_dim=fp_dim
        )

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            root_expansion_topk=root_expansion_topk,
            root_keep_first_reaction=root_keep_first_reaction
        )

    def plan_raw(self, target_mol, target_mol_id=0, viz=None, viz_dir=None,
                 banned_reactions=None, return_mol_tree=False,
                 root_rank_start=None, exclude_smiles=None):
        return self.plan_handle(
            target_mol,
            target_mol_id,
            viz_override=viz,
            viz_dir_override=viz_dir,
            banned_reactions=banned_reactions,
            return_mol_tree=return_mol_tree,
            root_rank_start=root_rank_start,
            exclude_smiles=exclude_smiles
        )

    def plan(self, target_mol):
        t0 = time.time()
        succ, msg = self.plan_raw(target_mol)
        return _format_plan_result(succ, msg, time.time() - t0, target_mol)


class RSPlannerInterRetro:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 include_bb=None,
                 exclude_bb=None,
                 interretro_root=None,
                 interretro_checkpoint=None,
                 localretro_root=None,
                 localretro_model_path=None,
                 localretro_config_path=None,
                 localretro_data_dir=None,
                 viz=True,
                 viz_dir='viz',
                 root_expansion_topk=None,
                 root_keep_first_reaction=False,
                 ):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)
        starting_mols = _apply_building_block_filters(
            starting_mols=starting_mols,
            include_bb=include_bb,
            exclude_bb=exclude_bb,
        )

        one_step = prepare_interretro(
            interretro_root=interretro_root,
            checkpoint_path=interretro_checkpoint,
            localretro_root=localretro_root,
            localretro_model_path=localretro_model_path,
            localretro_config_path=localretro_config_path,
            localretro_data_dir=localretro_data_dir,
            topk=expansion_topk,
            device=device
        )
        value_fn = one_step.value if use_value_fn else lambda x: 0.

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            root_expansion_topk=root_expansion_topk,
            root_keep_first_reaction=root_keep_first_reaction,
        )

    def plan_raw(self, target_mol, target_mol_id=0, viz=None, viz_dir=None,
                 banned_reactions=None, return_mol_tree=False,
                 root_rank_start=None, exclude_smiles=None):
        return self.plan_handle(
            target_mol,
            target_mol_id,
            viz_override=viz,
            viz_dir_override=viz_dir,
            banned_reactions=banned_reactions,
            return_mol_tree=return_mol_tree,
            root_rank_start=root_rank_start,
            exclude_smiles=exclude_smiles
        )

    def plan(self, target_mol):
        t0 = time.time()
        succ, msg = self.plan_raw(target_mol)
        return _format_plan_result(succ, msg, time.time() - t0, target_mol)


if __name__ == '__main__':
    planner = RSPlanner(
        gpu=0,
        use_value_fn=True,
        iterations=100,
        expansion_topk=50
    )

    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)

    result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    print(result)

    result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    print(result)
