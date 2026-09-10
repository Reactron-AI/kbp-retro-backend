import pickle
import pandas as pd
import logging
from mlp_retrosyn.mlp_inference import MLPModel
from retro_star.common.interretro_adapter import InterRetroAdapter
from retro_star.common.localretro_adapter import LocalRetroAdapter
from retro_star.alg import molstar


def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


def prepare_mlp(templates, model_dump):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=-1)
    return one_step


def prepare_localretro(localretro_root, model_path=None, config_path=None,
                       data_dir=None, device='cpu'):
    one_step = LocalRetroAdapter(
        localretro_root=localretro_root,
        model_path=model_path,
        config_path=config_path,
        data_dir=data_dir,
        device=device
    )
    return one_step


def prepare_interretro(interretro_root=None, checkpoint_path=None,
                       localretro_root=None, localretro_model_path=None,
                       localretro_config_path=None, localretro_data_dir=None,
                       topk=10, device='cpu'):
    one_step = InterRetroAdapter(
        interretro_root=interretro_root,
        checkpoint_path=checkpoint_path,
        localretro_root=localretro_root,
        localretro_model_path=localretro_model_path,
        localretro_config_path=localretro_config_path,
        localretro_data_dir=localretro_data_dir,
        topk=topk,
        device=device
    )
    return one_step


def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None,
                            root_expansion_topk=None,
                            root_keep_first_reaction=False):
    def plan_handle(x, y=0, viz_override=None, viz_dir_override=None,
                    banned_reactions=None, return_mol_tree=False,
                    root_rank_start=None, exclude_smiles=None,
                    exclude_smiles_strict=None):
        effective_viz = viz if viz_override is None else viz_override
        effective_viz_dir = viz_dir if viz_dir_override is None else viz_dir_override
        banned_count = len(banned_reactions or [])

        def expansion_handle(mol, depth=0):
            topk = expansion_topk
            if depth == 0 and root_expansion_topk is not None:
                if root_expansion_topk == 'all':
                    topk = None
                elif root_expansion_topk == 'next_unbanned':
                    topk = expansion_topk + banned_count
                elif root_expansion_topk == 'rank_after_banned':
                    if hasattr(one_step, 'run_rank_slice'):
                        return one_step.run_rank_slice(
                            mol,
                            start_rank=(
                                root_rank_start
                                if root_rank_start is not None
                                else banned_count
                            ),
                            limit=1
                        )
                    topk = banned_count + 1
                else:
                    topk = root_expansion_topk
            return one_step.run(mol, topk=topk)

        return molstar(
            target_mol=x,
            target_mol_id=y,
            starting_mols=starting_mols,
            expand_fn=expansion_handle,
            value_fn=value_fn,
            iterations=iterations,
            viz=effective_viz,
            viz_dir=effective_viz_dir,
            banned_reactions=banned_reactions,
            return_mol_tree=return_mol_tree,
            root_keep_first_reaction=root_keep_first_reaction,
            exclude_smiles=exclude_smiles,
            exclude_smiles_strict=exclude_smiles_strict
        )

    return plan_handle
