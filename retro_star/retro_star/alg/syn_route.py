import numpy as np
from queue import Queue
import logging
import os
from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound
from retro_star.route_utils import (
    ROUTE_MOL_IMAGE_SIZE, ROUTE_MOL_NODE_HEIGHT, ROUTE_MOL_NODE_WIDTH,
    route_node_attrs, route_reaction_children
)


class SynRoute:
    def __init__(self, target_mol, succ_value, search_status):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.template_ranks = [None]
        self.parents = [-1]
        self.children = [None]
        self.optimal = False
        self.costs = {}

        self.succ_value = succ_value
        self.total_cost = 0
        self.length = 0
        self.search_status = search_status
        if self.succ_value <= self.search_status:
            self.optimal = True

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.template_ranks.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, reactants, cost,
                     template_rank=None):
        assert mol in self.mols

        self.total_cost += cost
        self.length += 1

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.template_ranks[parent_id] = template_rank
        self.children[parent_id] = []
        self.costs[parent_id] = cost

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('graph', dpi='300')
        G.attr('graph', rankdir='RL')
        G.attr('graph', ordering='out')
        G.attr('node', shape='box')

        names = []
        for i in range(len(self.mols)):
            names.append('%d | %s' % (i, self.mols[i]))

        use_mol_images = False
        smiles_to_img = {}
        img_dir = '%s_mol_imgs' % viz_file
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            os.makedirs(img_dir, exist_ok=True)
            use_mol_images = True
            for idx, smiles in enumerate(self.mols):
                if smiles in smiles_to_img:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                img_path = os.path.abspath(os.path.join(img_dir, 'mol_%d.png' % idx))
                Draw.MolToFile(mol, img_path, size=ROUTE_MOL_IMAGE_SIZE)
                smiles_to_img[smiles] = img_path
        except ImportError:
            logging.warning('RDKit not available. Falling back to SMILES labels for route render.')

        for idx, smiles in enumerate(self.mols):
            node_attrs = route_node_attrs(self, idx)
            if use_mol_images and smiles in smiles_to_img:
                G.node(
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
                G.node(names[idx], label=smiles, **node_attrs)

        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], dir='back')

            if self.children[idx] is not None:
                for c in route_reaction_children(self, idx):
                    node_queue.put((c, idx))

        for output_format in ('pdf', 'png'):
            try:
                G.render(format=output_format)
            except ExecutableNotFound:
                logging.warning('Skipping route render: Graphviz "dot" executable not found on PATH.')
                break

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s
        s += '>%.4f>' % np.exp(-self.costs[idx])
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s
