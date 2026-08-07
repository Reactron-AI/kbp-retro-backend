import json
import math
import os
import tempfile

from graphviz import Digraph, Source
from graphviz.backend.execute import ExecutableNotFound

ROUTE_NODE_PENWIDTH = '7'
ROUTE_MOL_IMAGE_SIZE = (675, 450)
ROUTE_MOL_NODE_WIDTH = '3.0'
ROUTE_MOL_NODE_HEIGHT = '2.0'
ROUTE_TARGET_ATTRS = {
    'color': 'red',
    'fillcolor': '#ffdddd',
    'style': 'filled',
    'penwidth': ROUTE_NODE_PENWIDTH,
}
ROUTE_STARTING_BLOCK_ATTRS = {
    'color': 'green',
    'fillcolor': '#ddffdd',
    'style': 'filled',
    'penwidth': ROUTE_NODE_PENWIDTH,
}
ROUTE_INTERMEDIATE_ATTRS = {
    'color': 'orange',
    'fillcolor': '#ffe2b3',
    'style': 'filled',
    'penwidth': ROUTE_NODE_PENWIDTH,
}


def route_node_attrs(route, idx):
    if idx == 0:
        return ROUTE_TARGET_ATTRS
    if route.children[idx] is None:
        return ROUTE_STARTING_BLOCK_ATTRS
    return ROUTE_INTERMEDIATE_ATTRS


def route_reaction_children(route, idx):
    children = route.children[idx]
    if children is None:
        return []

    return sorted(
        children,
        key=lambda child_idx: (
            route.children[child_idx] is not None,
            route.mols[child_idx]
        )
    )


def canonicalize_smiles(smiles):
    try:
        from rdkit import Chem
    except ImportError:
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return smiles
    if mol is None:
        return smiles

    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)


def reaction_key(product, reactants):
    return (
        canonicalize_smiles(product),
        tuple(sorted(canonicalize_smiles(reactant)
                     for reactant in reactants if reactant))
    )


def reaction_key_to_dict(key):
    product, reactants = key
    return {
        'product': product,
        'reactants': list(reactants)
    }


def signature_to_report(signature):
    return [
        reaction_key_to_dict(key)
        for key in signature
    ]


def depth0_reaction_signature(steps):
    for step in steps:
        if step['depth'] == 0:
            return ((
                step['reaction_key']['product'],
                tuple(step['reaction_key']['reactants'])
            ),)
    return tuple()


def canonical_smiles_set(smiles_list):
    if not smiles_list:
        return set()

    return {
        canonicalize_smiles(smiles)
        for smiles in smiles_list
        if smiles
    }


def route_terminal_building_blocks(route):
    return sorted({
        canonicalize_smiles(mol)
        for idx, mol in enumerate(route.mols)
        if route.children[idx] is None
    })


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def route_steps(route):
    steps = []
    depth_by_mol = {0: 0}
    depth_counts = {}
    queue = [0]

    while queue:
        mol_idx = queue.pop(0)
        children = route.children[mol_idx]
        if children is None:
            continue

        depth = depth_by_mol[mol_idx]
        rxn_index = depth_counts.get(depth, 0)
        depth_counts[depth] = rxn_index + 1
        ordered_children = route_reaction_children(route, mol_idx)
        reactants = [route.mols[child_idx] for child_idx in ordered_children]
        key = reaction_key(route.mols[mol_idx], reactants)

        steps.append({
            'step': len(steps),
            'rxn_index': rxn_index,
            'depth': depth,
            'product': route.mols[mol_idx],
            'reactants': reactants,
            'reaction_key': reaction_key_to_dict(key),
            'template': json_safe(route.templates[mol_idx]),
            'template_rank': (
                route.template_ranks[mol_idx]
                if hasattr(route, 'template_ranks') else None
            ),
            'score': float(math.exp(-route.costs[mol_idx])),
            'cost': float(route.costs[mol_idx]),
            'is_terminal': all(route.children[child_idx] is None
                               for child_idx in children)
        })

        for child_idx in ordered_children:
            depth_by_mol[child_idx] = depth + 1
            queue.append(child_idx)

    return steps


def route_tree_segments(steps):
    segments = []
    for step in steps:
        if 'reaction_key' in step:
            reaction = step['reaction_key']
            segments.append('%s>>%s' % (
                reaction['product'],
                '.'.join(reaction['reactants'])
            ))
        else:
            segments.append('step=%s rxn=%s' % (
                step.get('step'),
                step.get('rxn_index')
            ))
    return segments


def route_dot_source(route, image_dir=None):
    graph = Digraph('G')
    graph.attr('graph', dpi='300')
    graph.attr('graph', rankdir='RL')
    graph.attr('graph', ordering='out')
    graph.attr('node', shape='box')

    names = []
    for idx, smiles in enumerate(route.mols):
        name = '%d | %s' % (idx, smiles)
        names.append(name)

    smiles_to_img = {}
    if image_dir is not None:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            os.makedirs(image_dir, exist_ok=True)
            for idx, smiles in enumerate(route.mols):
                if smiles in smiles_to_img:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                img_path = os.path.abspath(os.path.join(image_dir, 'mol_%d.png' % idx))
                Draw.MolToFile(mol, img_path, size=ROUTE_MOL_IMAGE_SIZE)
                smiles_to_img[smiles] = img_path
        except ImportError:
            smiles_to_img = {}

    for idx, smiles in enumerate(route.mols):
        node_attrs = route_node_attrs(route, idx)
        if smiles in smiles_to_img:
            graph.node(
                names[idx],
                label='',
                image=smiles_to_img[smiles],
                imagescale='true',
                fixedsize='true',
                width=ROUTE_MOL_NODE_WIDTH,
                height=ROUTE_MOL_NODE_HEIGHT,
                **node_attrs
            )
        else:
            graph.node(names[idx], label=smiles, **node_attrs)

    queue = [(0, -1)]
    while queue:
        idx, parent_idx = queue.pop(0)
        if parent_idx >= 0:
            graph.edge(names[parent_idx], names[idx], dir='back')

        children = route.children[idx]
        if children is not None:
            for child_idx in route_reaction_children(route, idx):
                queue.append((child_idx, idx))

    return graph.source


def write_route_artifacts(route, route_file, png_file):
    os.makedirs(os.path.dirname(route_file), exist_ok=True)
    os.makedirs(os.path.dirname(png_file), exist_ok=True)

    source = route_dot_source(route)
    with open(route_file, 'w') as f:
        f.write(source)
        f.write('\n')

    try:
        with tempfile.TemporaryDirectory() as image_dir:
            png_source = route_dot_source(route, image_dir=image_dir)
            png_bytes = Source(png_source).pipe(format='png')
    except ExecutableNotFound:
        return False

    with open(png_file, 'wb') as f:
        f.write(png_bytes)
    return True
