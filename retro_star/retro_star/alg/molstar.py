import os
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree
from retro_star.route_utils import reaction_key


def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None, banned_reactions=None,
            return_mol_tree=False, root_keep_first_reaction=False,
            exclude_smiles=None, exclude_smiles_strict=None):
    banned_reactions = set(banned_reactions or [])
    strict_banned_mols = set(exclude_smiles_strict or [])
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn,
        # exclude_smiles_strict은 재고 취급도 막아야 하므로 exclude_smiles와 합쳐서 전달
        exclude_mols=set(exclude_smiles or []) | strict_banned_mols
    )
    mol_tree.root_expansion_metadata = {}

    i = -1

    if not mol_tree.succ:
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            depth = 0 if m_next.parent is None else 1
            result = expand_fn(m_next.mol, depth=depth)
            if depth == 0 and result is not None:
                mol_tree.root_expansion_metadata = {
                    key: result[key]
                    for key in (
                        'root_template_rank',
                        'root_action_rank',
                        'root_rank_start',
                        'root_rank_exhausted',
                        'next_root_rank_start'
                    )
                    if key in result
                }

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']

                template_ranks = result.get('template_ranks')

                reactant_lists = []
                kept_costs = []
                kept_templates = []
                kept_template_ranks = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    if reaction_key(m_next.mol, reactant_list) in banned_reactions:
                        continue
                    # exclude_smiles_strict: 재고 취급 여부와 무관하게, 이 분자를
                    # 중간체(reactant)로도 만들어내는 반응 자체를 후보에서 제외한다.
                    if strict_banned_mols and strict_banned_mols.intersection(reactant_list):
                        continue
                    reactant_lists.append(reactant_list)
                    kept_costs.append(costs[j])
                    kept_templates.append(templates[j])
                    kept_template_ranks.append(
                        template_ranks[j] if template_ranks is not None else None
                    )
                    if depth == 0 and root_keep_first_reaction:
                        break

                assert m_next.open
                if len(reactant_lists) == 0:
                    succ = mol_tree.expand(m_next, None, None, None)
                else:
                    succ = mol_tree.expand(
                        m_next,
                        reactant_lists,
                        np.array(kept_costs),
                        kept_templates,
                        kept_template_ranks
                    )

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    if return_mol_tree:
        return mol_tree.succ, (best_route, i+1, mol_tree)

    return mol_tree.succ, (best_route, i+1)
