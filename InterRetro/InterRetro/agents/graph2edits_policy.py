import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy

from rdkit import Chem

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from models import DMPNNValue
from Graph2Edits.prepare_data import apply_edit_to_mol
from Graph2Edits.preprocess import check_edits
from Graph2Edits.models import Graph2Edits, BeamSearch
from Graph2Edits.models.model_utils import get_seq_edit_accuracy
from Graph2Edits.utils.generate_edits import generate_reaction_edits
from Graph2Edits.utils.rxn_graphs import MolGraph, RxnGraph, Vocab
from Graph2Edits.utils.collate_fn import get_batch_graphs, prepare_edit_labels

from rxnmapper import RXNMapper

import sys
import joblib
import os

from agents.base_agent import BaseAgent


class Graph2EditsPolicy(BaseAgent):
    def __init__(self, args):
        self.args = args
        self.updates = 0

        # self.rxn_mapper = RXNMapper()
        self.bond_vocab = Vocab(joblib.load(self.args.bond_vocab_file))
        self.atom_vocab = Vocab(joblib.load(self.args.atom_vocab_file))

        # Load Graph2Edits
        checkpoint = torch.load(args.graph2edits_model, map_location=torch.device('cpu'), weights_only=False)
        config = checkpoint['saveables']
        self.graph2edits = Graph2Edits(**config, device=args.device)
        self.graph2edits.load_state_dict(checkpoint['state'])
        self.graph2edits.to(args.device)
        self.graph2edits.eval()
        self.graph2edits_optim = torch.optim.Adam(self.graph2edits.parameters(), lr=args.lr)

        # Load bream search
        self.beam_model = BeamSearch(model=self.graph2edits, step_beam_size=5,
                                     beam_size=10, use_rxn_class=False)

        # Prepare value function
        checkpoint = torch.load(args.graph2edits_model, map_location=torch.device('cpu'), weights_only=False)
        config = checkpoint['saveables']
        self.value_fn = DMPNNValue(**config, device=args.device)
        self.value_fn.load_state_dict(checkpoint['state'])
        self.value_fn.to(args.device)
        self.value_fn.eval()
        self.value_optim = torch.optim.Adam(self.value_fn.parameters(), lr=args.lr)

        self.nn_models = {'policy': self.graph2edits,
                          'value_fn': self.value_fn}

    def run(self, p_smi, topk=5):
        p_smi = self.canonicalize(p_smi)
        mol = Chem.MolFromSmiles(p_smi)
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        labeled_smi = Chem.MolToSmiles(mol)
        top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=max(5, topk), rxn_class=False)

        all_reactants = []
        all_edits = []
        for res in top_k_results:
            if not res['final_smi'].startswith('final_smi_unmapped'):
                unlabelled_smi = self.canonicalize(res['final_smi'])
                all_reactants.append(unlabelled_smi)

                edit_seq_with_params = []
                for action, atom_or_atoms in zip(res['edits'], res['edits_atom']):
                    edit_seq_with_params.append((action, atom_or_atoms))
                edit_seq_with_params.append(('Terminate', None))
                all_edits.append(edit_seq_with_params)
        return {'reactants': all_reactants[:topk],
                'scores': [1 for i in range(len(all_reactants))][:topk],
                'template': all_edits[:topk],
                'scores_reference': [1 for i in range(len(all_reactants))][:topk]}

    def compute_react_values(self, react_smiles):
        react_values_list = []
        for reacts in react_smiles:
            if len(reacts) > 1:
                # Multiple reactant SMILES
                val_batch = self.value_fn.forward_with_smiles(reacts)
                min_val = val_batch.min()
                react_values_list.append(min_val)
            else:
                # Single reactant SMILES
                val_single = self.value_fn.forward_with_smiles(reacts)
                react_values_list.append(val_single.squeeze())
        react_values = torch.stack(react_values_list).to(device=self.args.device).unsqueeze(dim=-1)
        return react_values

    def fit_value(self, rxn_batch):
        prod_smiles = rxn_batch['obs']
        react_smiles = rxn_batch['next_obs']
        rewards = rxn_batch['rewards']

        prod_values = self.value_fn.forward_with_smiles(prod_smiles)
        with torch.no_grad():
            react_values = self.compute_react_values(react_smiles)

            rewards_list = []
            for rwds in rewards:
                rewards_list.append(int(len(rwds) == sum(rwds)))
            rewards_tensor = torch.tensor(rewards_list).to(device=self.args.device).unsqueeze(dim=-1)

            targets = rewards_tensor + 0.98 * (1 - rewards_tensor) * react_values
            targets = torch.clamp(targets, min=0, max=1.0)

        loss = F.mse_loss(prod_values, targets)
        self.value_optim.zero_grad()
        loss.backward()
        self.value_optim.step()

        return {'value_fn/training_loss': loss.item(),
                'value_fn/target': targets.mean().item()}

    def fit_policy(self, rxn_batch):

        # ----------------------------------------------------
        # 1) Compute the advantage for each trajectory in the batch
        # ----------------------------------------------------
        prod_smiles = rxn_batch['obs']
        react_smiles = rxn_batch['next_obs']
        rewards = rxn_batch['rewards']

        prod_values = self.value_fn.forward_with_smiles(prod_smiles)  # shape: (batch_size, 1)
        with torch.no_grad():
            react_values = self.compute_react_values(react_smiles)

            rewards_list = []
            for rwds in rewards:
                rewards_list.append(int(len(rwds) == sum(rwds)))
            rewards_tensor = torch.tensor(rewards_list).to(device=self.args.device).unsqueeze(dim=-1)

            targets = rewards_tensor + 0.98 * (1 - rewards_tensor) * react_values
            advantage = targets - prod_values
            advantage = advantage.squeeze(dim=-1)  # shape: (batch_size,)
            advantage = torch.clamp(advantage, min=-1, max=1)

            weights = torch.exp(self.args.adv_coef * advantage).detach()  # shape: (batch_size,)
            weights = torch.clamp(weights, min=0, max=100).detach()

        # ----------------------------------------------------
        # 2) Forward pass through Graph2Edits to get predictions
        # ----------------------------------------------------
        batch_graphs = rxn_batch['graphs']
        graph_seq_tensors, seq_labels, seq_mask = self.process_batch(batch_graphs)
        seq_edit_scores = self.graph2edits(graph_seq_tensors)
        max_seq_len, batch_size = seq_mask.size()

        # ----------------------------------------------------
        # 3) Weighted cross-entropy for each step
        # ----------------------------------------------------
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        seq_loss = []
        for step_idx in range(max_seq_len):
            target_labels = self.graph2edits.to_device(seq_labels[step_idx])
            # Sum the cross-entropy over the batch, but only for active steps
            loss_batch = []
            for i in range(batch_size):
                if seq_mask[step_idx][i] == 1:  # valid step
                    pred_logits = seq_edit_scores[step_idx][i].unsqueeze(0)
                    gold_label = torch.argmax(target_labels[i]).unsqueeze(0).long()
                    # loss_batch.append(loss_fn(pred_logits, gold_label).sum())

                    ce_i = loss_fn(pred_logits, gold_label).sum()
                    w_i = weights[i]
                    loss_batch.append(w_i * ce_i)

            if len(loss_batch) > 0:
                seq_loss.append(torch.stack(loss_batch).mean())

        if len(seq_loss) == 0:
            return {}

        total_loss = torch.stack(seq_loss).mean()

        # ----------------------------------------------------
        # 4) Backprop and update
        # ----------------------------------------------------
        self.graph2edits_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.graph2edits.parameters(), 10)
        self.graph2edits_optim.step()

        accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)

        return {
            'graph2edits/training_loss': total_loss.item(),
            'graph2edits/training_accuracy': accuracy,
            'graph2edits/mean_advantage': advantage.mean().item(),
            'graph2edits/mean_weight': weights.mean().item()
        }

    def build_graph_seq(self, product_smi, reactants_smi, edit_seq):
        """
        product_smi:    e.g. 'CCC(=O)N'
        reactants_smi:  e.g. 'CC.CN' or 'CC.O' or single 'CC'
        edit_seq:       a list of ((actionType, actionParams), [atomMap(s)]) steps.
                        e.g. [ (('Delete Bond',(None,None)), [1,2]),
                               (('Change Atom',(2,0)), [3]),
                               ('Terminate', None) ]
        Returns: a list of RxnGraph objects, each describing one step in the transformation.
        """
        try:
            r_mol = Chem.MolFromSmiles(reactants_smi)
            p_mol = Chem.MolFromSmiles(product_smi)
            Chem.Kekulize(p_mol)
        except:
            return []  # can't parse?

        graph_seq = []
        int_mol = p_mol
        for (edit_action, edit_atoms) in edit_seq:
            # Create the RxnGraph for this step
            rxn_graph = RxnGraph(
                prod_mol=Chem.Mol(int_mol),
                edit_to_apply=edit_action,
                edit_atom=edit_atoms,
                reac_mol=Chem.Mol(r_mol),
                rxn_class=None,
                use_rxn_class=False
            )
            graph_seq.append(rxn_graph)

            # If not 'Terminate', apply the edit to int_mol
            if edit_action == 'Terminate':
                # after Termination, we just break or let it continue with a None check
                pass
            else:
                int_mol = apply_edit_to_mol(
                    Chem.Mol(int_mol), edit_action, edit_atoms
                )
                if int_mol is None:
                    print("Warning: applying edit resulted in None molecule. Stopping.")
                    break

        return graph_seq

    def process_batch(self, batch_graphs):
        lengths = torch.tensor([len(graph_seq)
                                for graph_seq in batch_graphs], dtype=torch.long)
        max_length = max([len(graph_seq) for graph_seq in batch_graphs])

        graph_seq_tensors = []
        edit_seq_labels = []
        seq_mask = []

        for idx in range(max_length):
            graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length - 1)]).get_components(
                attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                          for i, length in enumerate(lengths)]
            mask = (idx < lengths).long()
            prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))
            assert all([isinstance(graph, MolGraph) for graph in prod_graphs])

            edit_labels = prepare_edit_labels(
                prod_graphs, edits, edit_atoms, self.bond_vocab, self.atom_vocab)
            current_graph_tensors = get_batch_graphs(
                prod_graphs, use_rxn_class=None)

            graph_seq_tensors.append(current_graph_tensors)
            edit_seq_labels.append(edit_labels)
            seq_mask.append(mask)

        seq_mask = torch.stack(seq_mask).long()
        assert seq_mask.shape[0] == max_length
        assert seq_mask.shape[1] == len(batch_graphs)

        return graph_seq_tensors, edit_seq_labels, seq_mask

    def preprocessing(self, rxns):
        rxns_data = []
        counter = []
        all_edits = {}

        for idx, rxn_smi in enumerate(rxns):
            r, p = rxn_smi.split('>>')
            prod_mol = Chem.MolFromSmiles(p)

            if (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
                print(
                    f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

            react_mol = Chem.MolFromSmiles(r)

            if (react_mol is None) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
                print(
                    f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

            try:
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=True)
            except:
                print(f'Failed to extract reaction data, skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

            edits_accepted = check_edits(rxn_data.edits)
            if not edits_accepted:
                print(f'Edit: Add new bond. Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

            rxns_data.append(rxn_data)

        sys.stdout.flush()

        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit not in all_edits:
                    all_edits[edit] = 1
                else:
                    all_edits[edit] += 1

        atom_edits = []
        bond_edits = []
        lg_edits = []
        atom_lg_edits = []

        for edit, num in all_edits.items():
            if edit[0] == 'Change Atom':
                atom_edits.append(edit)
                atom_lg_edits.append(edit)
            elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                bond_edits.append(edit)
            elif edit[0] == 'Attaching LG':
                lg_edits.append(edit)
        atom_lg_edits.extend(lg_edits)

        filter_rxns_data = []
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit[0] == 'Attaching LG' and edit not in lg_edits:
                    print(
                        f'The number of {edit} in training set is very small, skipping reaction')
                    rxn_data = None
            if rxn_data is not None:
                counter.append(len(rxn_data.edits))
                filter_rxns_data.append(rxn_data)

        return filter_rxns_data

    def get_reactants(self, p_smi, sampling=False, max_steps=5):
        p_smi = self.canonicalize(p_smi)
        mol = Chem.MolFromSmiles(p_smi)
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        labeled_smi = Chem.MolToSmiles(mol)
        top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=max_steps, rxn_class=False)
        all_reactants = []
        all_reactant_probs = []
        all_edits = []
        for res in top_k_results:
            if not res['final_smi'].startswith('final_smi_unmapped'):
                reactant_list = res['final_smi'].split('.')
                all_reactants.append(reactant_list)
                all_reactant_probs.append(res['prob'])

                edit_seq_with_params = []
                for action, atom_or_atoms in zip(res['edits'], res['edits_atom']):
                    edit_seq_with_params.append((action, atom_or_atoms))
                edit_seq_with_params.append(('Terminate', None))
                all_edits.append(edit_seq_with_params)

        if len(all_reactants) == 0:
            raise ValueError(f"Final SMILES for {p_smi} are unmapped.")

        if sampling:
            logits = np.log(np.array(all_reactant_probs) + 1e-8)
            adjusted_probs = np.exp(logits / self.args.temperature)
            adjusted_probs /= adjusted_probs.sum()

            idx = random.choices(range(len(all_reactants)), weights=adjusted_probs, k=1)[0]
            return all_reactants[idx], all_edits[idx]
        else:
            return all_reactants[0], all_edits[0]

    def canonicalize(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            print('no mol', flush=True)
            return smi
        if mol is None:
            return smi
        mol = Chem.RemoveHs(mol)
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)