"""
Multi-route case-study search: repeatedly calls a planner with banned_reactions
to collect several distinct (unique first-step) synthesis routes for one target,
writing route.txt / route.png artifacts per route.

Shared by retro_star/example.py (CLI) and backend/retro_wrapper/planner.py (API),
so both produce identical route output instead of two diverging implementations.
"""

import os
import time

from retro_star.route_utils import (
    depth0_reaction_signature,
    json_safe,
    route_steps,
    route_tree_segments,
    signature_to_report,
    write_route_artifacts,
)


def smiles_ring_count(smiles):
    try:
        from rdkit import Chem
    except ImportError:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    return int(mol.GetRingInfo().NumRings())


def depth0_ring_counts(steps):
    for step in steps:
        if step['depth'] != 0:
            continue

        product_rings = smiles_ring_count(step['product'])
        reactant_ring_counts = [
            smiles_ring_count(reactant)
            for reactant in step['reactants']
        ]
        reactants_total_rings = None
        if all(count is not None for count in reactant_ring_counts):
            reactants_total_rings = sum(reactant_ring_counts)

        changed = (
            product_rings is not None and
            reactants_total_rings is not None and
            product_rings > reactants_total_rings
        )
        return {
            'product_total_rings': product_rings,
            'reactant_ring_counts': reactant_ring_counts,
            'reactants_total_rings': reactants_total_rings,
            'changed': changed,
        }

    return {
        'product_total_rings': None,
        'reactant_ring_counts': [],
        'reactants_total_rings': None,
        'changed': False,
    }


def mol_tree_depth0_reaction_signature(mol_tree):
    if mol_tree is None or not mol_tree.root.children:
        return tuple()

    reaction = mol_tree.root.children[0]
    reactants = [child.mol for child in reaction.children]
    from retro_star.route_utils import reaction_key
    return (reaction_key(mol_tree.root.mol, reactants),)


def case_study_route_path(case_study_dir, route_idx, prefix=''):
    return os.path.join(
        case_study_dir,
        'routes',
        '%smol_%d_route.txt' % (prefix, route_idx)
    )


def case_study_viz_path(case_study_dir, route_idx, prefix=''):
    return os.path.join(
        case_study_dir,
        'viz',
        '%smol_%d_route.png' % (prefix, route_idx)
    )


def case_study_result_path(case_study_dir):
    return os.path.join(case_study_dir, 'routes.json')


def collect_unique_depth0_routes(planner, target_mol, case_study_dir,
                                  planner_name='', max_routes=100,
                                  write_artifacts=True, exclude_smiles=None):
    """
    Search for up to `max_routes` synthesis routes for `target_mol`, treating
    routes as distinct only if their first (depth-0) reaction differs. Writes
    routes/route.txt and viz/route.png per route under `case_study_dir` when
    write_artifacts is True.

    Returns the same result dict shape that retro_star/example.py writes to
    routes.json.
    """
    routes_dir = os.path.join(case_study_dir, 'routes')
    viz_dir = os.path.join(case_study_dir, 'viz')
    if write_artifacts:
        os.makedirs(routes_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

    banned_reactions = set()
    depth0_signatures = []
    failed_depth0_signatures = []
    routes = []
    skipped_duplicate_root_routes = 0
    skipped_failed_root_routes = 0
    skipped_empty_root_ranks = 0
    root_rank_start = 0
    stop_reason = 'max_routes'
    t0 = time.time()

    search_idx = 0
    while len(routes) < max_routes:
        current_search_idx = search_idx
        search_idx += 1
        succ, msg = planner.plan_raw(
            target_mol,
            target_mol_id=current_search_idx,
            viz=False,
            viz_dir=None,
            banned_reactions=banned_reactions,
            return_mol_tree=True,
            root_rank_start=root_rank_start,
            exclude_smiles=exclude_smiles
        )
        mol_tree = msg[2]
        root_expansion_metadata = getattr(
            mol_tree,
            'root_expansion_metadata',
            {}
        )
        if 'next_root_rank_start' in root_expansion_metadata:
            root_rank_start = root_expansion_metadata['next_root_rank_start']

        if not succ:
            failed_signature = mol_tree_depth0_reaction_signature(mol_tree)
            if failed_signature:
                banned_reactions.update(failed_signature)
                failed_depth0_signatures.append(failed_signature)
                skipped_failed_root_routes += 1
                continue

            if root_expansion_metadata and not root_expansion_metadata.get(
                    'root_rank_exhausted', False):
                skipped_empty_root_ranks += 1
                continue

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
        ring_counts = depth0_ring_counts(steps)
        artifact_prefix = '#ring_' if ring_counts['changed'] else ''
        route_file = case_study_route_path(
            case_study_dir,
            current_search_idx,
            artifact_prefix
        )
        route_viz = case_study_viz_path(
            case_study_dir,
            current_search_idx,
            artifact_prefix
        )

        viz_written = False
        if write_artifacts:
            viz_written = write_route_artifacts(route, route_file, route_viz)

        routes.append({
            'route_index': route_idx,
            'search_index': current_search_idx,
            'succ': True,
            'iter': iteration_count,
            'route_file': route_file if write_artifacts else None,
            'route_viz': route_viz if viz_written else None,
            'route': route.serialize(),
            'route_tree_segments': route_tree_segments(steps),
            'steps': steps,
            'depth0_reaction_signature': signature_to_report(depth0_signature),
            'depth0_ring_counts': ring_counts,
            'root_expansion_metadata': json_safe(root_expansion_metadata),
            'route_cost': float(route.total_cost),
            'route_len': route.length,
            'optimal': route.optimal,
        })

    result = {
        'target_mol': target_mol,
        'planner': planner_name,
        'novelty_policy': 'depth0_unique_reactions',
        'max_routes': max_routes,
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
        'failed_depth0_reaction_signatures': [
            signature_to_report(signature)
            for signature in failed_depth0_signatures
        ],
        'skipped_duplicate_root_routes': skipped_duplicate_root_routes,
        'skipped_failed_root_routes': skipped_failed_root_routes,
        'skipped_empty_root_ranks': skipped_empty_root_ranks,
        'next_root_rank_start': root_rank_start,
        'stop_reason': stop_reason,
    }

    return result
