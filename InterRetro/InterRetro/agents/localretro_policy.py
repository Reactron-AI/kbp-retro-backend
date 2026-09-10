import importlib
import logging
import os
import random
import sys
from collections import OrderedDict

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

from agents.base_agent import BaseAgent


def _default_localretro_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'LocalRetro')
    )


def _default_localretro_paths(localretro_root):
    return {
        'model_path': os.path.join(localretro_root, 'models', 'LocalRetro_Pistachio_epoch_45.pth'),
        'config_path': os.path.join(localretro_root, 'data', 'configs', 'default_config.json'),
        'data_dir': os.path.join(localretro_root, 'data', 'Pistachio')
    }


def _ensure_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{description} not found: {path}')


class LocalRetroPolicy(BaseAgent):
    """Adapter that makes LocalRetro look like an InterRetro one-step agent."""

    def __init__(self, args):
        self.args = args
        self.is_trainable = True

        self.localretro_root = os.path.abspath(args.localretro_root or _default_localretro_root())
        defaults = _default_localretro_paths(self.localretro_root)
        self.model_path = args.localretro_model_path or defaults['model_path']
        self.config_path = args.localretro_config_path or defaults['config_path']
        self.data_dir = args.localretro_data_dir or defaults['data_dir']
        self.topk = args.localretro_topk
        self.policy_batch_size = max(1, getattr(args, 'localretro_policy_batch_size', 1))
        self.gamma = 0.98
        self.fp_dim = getattr(args, 'value_fp_dim', 2048)

        _ensure_exists(self.localretro_root, 'LocalRetro root')
        _ensure_exists(self.model_path, 'LocalRetro model checkpoint')
        _ensure_exists(self.config_path, 'LocalRetro config')
        _ensure_exists(self.data_dir, 'LocalRetro data directory')

        if self.localretro_root not in sys.path:
            sys.path.insert(0, self.localretro_root)

        try:
            localretro_module = importlib.import_module('Retrosynthesis')
            get_edit_module = importlib.import_module('scripts.get_edit')
            template_decoder_module = importlib.import_module('LocalTemplate.template_decoder')
        except ImportError as exc:
            raise ImportError(
                f'Failed to import LocalRetro from {self.localretro_root}. '
                'Make sure the LocalRetro checkout and dependencies are available.'
            ) from exc
        self.combined_edit = get_edit_module.combined_edit
        self.read_prediction = template_decoder_module.read_prediction
        self.decode_localtemplate = template_decoder_module.decode_localtemplate

        logging.info('Loading LocalRetro one-step model from %s', self.localretro_root)
        logging.info('LocalRetro checkpoint: %s', self.model_path)
        logging.info('LocalRetro config: %s', self.config_path)
        logging.info('LocalRetro data dir: %s', self.data_dir)

        self.localretro = localretro_module.LocalRetro({
            'model_path': self.model_path,
            'config_path': self.config_path,
            'data_dir': self.data_dir,
            'device': self.args.device,
        })
        self.localretro.model.to(self.args.device)
        self.localretro.model.eval()

        self.value_fn = ValueMLP(
            n_layers=getattr(args, 'value_n_layers', 1),
            fp_dim=self.fp_dim,
            latent_dim=getattr(args, 'value_latent_dim', 128),
            dropout_rate=getattr(args, 'value_dropout', 0.1),
        ).to(self.args.device)
        self.value_fn.eval()

        self.policy_optim = torch.optim.Adam(self.localretro.model.parameters(), lr=args.lr)
        self.value_optim = torch.optim.Adam(self.value_fn.parameters(), lr=args.lr)
        self.nn_models = {
            'policy': self.localretro.model,
            'value_fn': self.value_fn,
        }

    def run(self, p_smi, topk=5):
        p_smi = self.canonicalize(p_smi)
        predictions = self._predict_decoded(p_smi, topk=topk)
        return self._format_predictions(p_smi, predictions, topk=topk)

    def _format_predictions(self, p_smi, predictions, topk=None, metadata=None):
        metadata = metadata or {}
        if len(predictions) == 0:
            result = {'reactants': [], 'scores': [], 'template': [], 'scores_reference': []}
            result.update(metadata)
            return result

        merged = OrderedDict()
        for prediction in predictions:
            reactant_parts = [self.canonicalize(part) for part in prediction['reactants'].split('.') if part]
            reactant_parts = sorted(part for part in reactant_parts if part)
            if len(reactant_parts) == 0:
                continue

            reactant_key = '.'.join(reactant_parts)
            if reactant_key == p_smi:
                continue

            if reactant_key not in merged:
                merged[reactant_key] = {
                    'score': prediction['score'],
                    'action': prediction['action'],
                    'best_action_score': prediction['score'],
                }
            else:
                merged[reactant_key]['score'] += prediction['score']
                if prediction['score'] > merged[reactant_key]['best_action_score']:
                    merged[reactant_key]['action'] = prediction['action']
                    merged[reactant_key]['best_action_score'] = prediction['score']

        ranked = sorted(merged.items(), key=lambda item: item[1]['score'], reverse=True)
        if topk is not None:
            ranked = ranked[:topk]
        if len(ranked) == 0:
            result = {'reactants': [], 'scores': [], 'template': [], 'scores_reference': []}
            result.update(metadata)
            return result

        total_score = sum(max(item[1]['score'], 0.0) for item in ranked)
        if total_score <= 0:
            scores = [1.0 / len(ranked)] * len(ranked)
        else:
            scores = [max(item[1]['score'], 0.0) / total_score for item in ranked]

        result = {
            'reactants': [item[0] for item in ranked],
            'scores': scores,
            'template': [item[1]['action'] for item in ranked],
            'scores_reference': scores,
        }
        result.update(metadata)
        return result

    def run_rank_slice(self, p_smi, start_rank=0, limit=1):
        p_smi = self.canonicalize(p_smi)
        end_rank = start_rank + limit
        raw_predictions = self._predict_raw(p_smi, end_rank)
        metadata = {
            'root_action_rank': start_rank,
            'root_rank_start': start_rank,
            'root_rank_exhausted': len(raw_predictions) <= start_rank,
            'next_root_rank_start': min(end_rank, len(raw_predictions))
        }
        if metadata['root_rank_exhausted']:
            return self._format_predictions(p_smi, [], topk=None, metadata=metadata)

        predictions = self._decode_raw_predictions(
            p_smi,
            raw_predictions[start_rank:end_rank]
        )
        return self._format_predictions(
            p_smi,
            predictions,
            topk=None,
            metadata=metadata
        )

    def get_reactants(self, p_smi, sampling=False, max_steps=5):
        topk = max(self.topk, max_steps)
        results = self.run(p_smi, topk=topk)
        reactants = results['reactants']
        if len(reactants) == 0:
            return [], None

        if sampling:
            probs = np.array(results['scores'], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            idx = random.choices(range(len(reactants)), weights=probs, k=1)[0]
        else:
            idx = 0

        return reactants[idx].split('.'), results['template'][idx]

    def fit_policy(self, rxn_batch):
        valid_items = self._prepare_policy_items(rxn_batch)
        if len(valid_items) == 0:
            return {}

        with torch.no_grad():
            product_values, react_values, rewards = self._batch_values_for_items(valid_items)
            targets = rewards + self.gamma * (1 - rewards) * react_values
            advantage = torch.clamp(targets - product_values, min=-1, max=1).squeeze(dim=-1)
            weights = torch.exp(self.args.adv_coef * advantage).detach()
            weights = torch.clamp(weights, min=0, max=100).detach()

        self.policy_optim.zero_grad()
        self.localretro.model.train()
        total_loss_value = 0.0
        n_loss_chunks = 0
        for start in range(0, len(valid_items), self.policy_batch_size):
            end = min(start + self.policy_batch_size, len(valid_items))
            chunk_items = valid_items[start:end]
            chunk_weights = weights[start:end]
            chunk_loss = self._policy_chunk_loss(chunk_items, chunk_weights)
            if chunk_loss is None:
                continue

            scaled_loss = chunk_loss * (len(chunk_items) / len(valid_items))
            scaled_loss.backward()
            total_loss_value += scaled_loss.item()
            n_loss_chunks += 1
            if torch.cuda.is_available() and str(self.args.device).startswith('cuda'):
                torch.cuda.empty_cache()

        if n_loss_chunks == 0:
            self.localretro.model.eval()
            return {}

        nn.utils.clip_grad_norm_(self.localretro.model.parameters(), 20)
        self.policy_optim.step()
        self.localretro.model.eval()

        return {
            'localretro_policy/training_loss': total_loss_value,
            'localretro_policy/mean_advantage': advantage.mean().item(),
            'localretro_policy/mean_weight': weights.mean().item(),
            'localretro_policy/microbatches': n_loss_chunks,
        }

    def _policy_chunk_loss(self, items, weights):
        graphs = [item['graph'] for item in items]
        atom_logits, bond_logits = self._forward_graphs(graphs)
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        atom_offset = 0
        bond_offset = 0
        sample_losses = []
        for idx, item in enumerate(items):
            graph = item['graph']
            action = item['action']
            num_atoms = graph.number_of_nodes()
            num_bonds = graph.remove_self_loop().number_of_edges()

            atom_labels = torch.zeros(num_atoms, dtype=torch.long, device=self.args.device)
            bond_labels = torch.zeros(num_bonds, dtype=torch.long, device=self.args.device)
            if action['edit_type'] == 'a':
                atom_labels[action['edit_site']] = action['template_class']
            else:
                bond_labels[action['edit_site']] = action['template_class']

            loss_parts = []
            if num_atoms > 0:
                atom_slice = atom_logits[atom_offset:atom_offset + num_atoms]
                loss_parts.append(loss_fn(atom_slice, atom_labels).mean())
            if num_bonds > 0:
                bond_slice = bond_logits[bond_offset:bond_offset + num_bonds]
                loss_parts.append(loss_fn(bond_slice, bond_labels).mean())
            if len(loss_parts) > 0:
                sample_losses.append(weights[idx] * torch.stack(loss_parts).mean())

            atom_offset += num_atoms
            bond_offset += num_bonds

        if len(sample_losses) == 0:
            return None
        return torch.stack(sample_losses).mean()

    def fit_value(self, rxn_batch):
        valid_items = self._prepare_value_items(rxn_batch)
        if len(valid_items) == 0:
            return {}

        self.value_fn.train()
        product_values, react_values, rewards = self._batch_values_for_items(valid_items)
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - rewards) * react_values
            targets = torch.clamp(targets, min=0, max=1.0)

        loss = F.mse_loss(product_values, targets)
        self.value_optim.zero_grad()
        loss.backward()
        self.value_optim.step()
        self.value_fn.eval()

        return {
            'localretro_value/training_loss': loss.item(),
            'localretro_value/target': targets.mean().item(),
        }

    def _predict_decoded(self, smiles, topk):
        return self._decode_raw_predictions(smiles, self._predict_raw(smiles, topk))

    def _decode_raw_predictions(self, smiles, raw_predictions):
        predictions = []
        for raw in raw_predictions:
            edit_type, edit_site, template_class, score = raw
            try:
                mol, pred_site, template, template_info, score = self.read_prediction(
                    smiles,
                    raw,
                    self.localretro.atom_templates,
                    self.localretro.bond_templates,
                    self.localretro.template_infos,
                    True,
                )
                local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
                decoded_smiles = self.decode_localtemplate(mol, pred_site, local_template, template_info)
            except Exception:
                continue
            if decoded_smiles is None:
                continue
            action = {
                'edit_type': edit_type,
                'edit_site': int(edit_site),
                'template_class': int(template_class),
                'template': local_template,
                'score': float(score),
                'product_smiles': smiles,
            }
            predictions.append({
                'reactants': decoded_smiles,
                'score': float(score),
                'action': action,
            })
        return predictions

    def _predict_raw(self, smiles, topk):
        self.localretro.model.eval()
        graph = self.localretro.graph_function(smiles)
        with torch.no_grad():
            atom_logits, bond_logits = self._forward_graphs([graph])
            atom_probs = nn.Softmax(dim=1)(atom_logits)
            bond_probs = nn.Softmax(dim=1)(bond_logits)
            if topk is None:
                atom_candidates = atom_probs.numel() - atom_probs.size(0)
                bond_candidates = bond_probs.numel() - bond_probs.size(0)
                topk = atom_candidates + bond_candidates
            pred_types, pred_sites, pred_scores = self.combined_edit(
                graph.remove_self_loop(),
                atom_probs,
                bond_probs,
                topk,
            )
        return [
            (pred_types[k], pred_sites[k][0], pred_sites[k][1], pred_scores[k])
            for k in range(len(pred_types))
        ]

    def _forward_graphs(self, graphs):
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg = bg.to(self.args.device)
        node_feats = bg.ndata.pop('h').to(self.args.device)
        edge_feats = bg.edata.pop('e').to(self.args.device)
        atom_logits, bond_logits, _ = self.localretro.model(bg, node_feats, edge_feats)
        return atom_logits, bond_logits

    def _prepare_policy_items(self, rxn_batch):
        items = []
        for obs, action, next_obs, rewards in zip(
            rxn_batch['obs'],
            rxn_batch['actions'],
            rxn_batch['next_obs'],
            rxn_batch['rewards'],
        ):
            if not isinstance(action, dict):
                continue
            product_smiles = action.get('product_smiles') or self.canonicalize(obs)
            try:
                graph = self.localretro.graph_function(product_smiles)
            except Exception:
                continue
            if not self._valid_action_for_graph(action, graph):
                continue
            items.append({
                'obs': product_smiles,
                'action': action,
                'next_obs': next_obs,
                'rewards': rewards,
                'graph': graph,
            })
        return items

    def _prepare_value_items(self, rxn_batch):
        items = []
        for obs, next_obs, rewards in zip(
            rxn_batch['obs'],
            rxn_batch['next_obs'],
            rxn_batch['rewards'],
        ):
            product_smiles = self.canonicalize(obs)
            if Chem.MolFromSmiles(product_smiles) is None:
                continue
            items.append({
                'obs': product_smiles,
                'next_obs': next_obs,
                'rewards': rewards,
            })
        return items

    def _valid_action_for_graph(self, action, graph):
        edit_type = action.get('edit_type')
        edit_site = action.get('edit_site')
        template_class = action.get('template_class')
        if edit_type not in ['a', 'b'] or edit_site is None or template_class is None:
            return False
        if template_class <= 0:
            return False
        if edit_type == 'a':
            return 0 <= edit_site < graph.number_of_nodes()
        return 0 <= edit_site < graph.remove_self_loop().number_of_edges()

    def _batch_values_for_items(self, items):
        product_smiles = [item['obs'] for item in items]
        product_values = self._value_for_smiles(product_smiles)
        react_values = []
        rewards = []
        for item in items:
            reactants = [self.canonicalize(smi) for smi in item['next_obs']]
            if len(reactants) == 0:
                react_values.append(torch.tensor(0.0, device=self.args.device))
            else:
                reactant_values = self._value_for_smiles(reactants).squeeze(dim=-1)
                react_values.append(reactant_values.min())
            reward = int(len(item['rewards']) == sum(item['rewards']))
            rewards.append(reward)
        react_values = torch.stack(react_values).unsqueeze(dim=-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.args.device).unsqueeze(dim=-1)
        return product_values, react_values, rewards

    def _value_for_smiles(self, smiles_list):
        fps = np.array([self.smiles_to_fp(smi) for smi in smiles_list], dtype=np.float32)
        fps = torch.tensor(fps, dtype=torch.float32, device=self.args.device)
        return self.value_fn(fps)

    def smiles_to_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fp_dim, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_dim)
        arr = np.zeros((self.fp_dim,), dtype=np.float32)
        arr[list(fp.GetOnBits())] = 1.0
        return arr

    def canonicalize(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            return smi
        if mol is None:
            return smi
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol)


class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        layers = [
            nn.Linear(fp_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
        layers.append(nn.Linear(latent_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = self.layers(fps)
        return torch.log(1 + torch.exp(x))
