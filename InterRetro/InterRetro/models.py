from typing import Dict, List, Tuple, Union

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from Graph2Edits.prepare_data import apply_edit_to_mol
from Graph2Edits.utils.collate_fn import get_batch_graphs
from Graph2Edits.utils.rxn_graphs import MolGraph, Vocab

from Graph2Edits.models.encoder import Global_Attention, MPNEncoder
from Graph2Edits.models.model_utils import (creat_edits_feats, index_select_ND,
                                            unbatch_feats)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict, OrderedDict
from rdchiral.main import rdchiralRunText

from interretro_utils import smiles2array

def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, srs, ts, ids = zip(*l)
    #     k = np.argmax(ss)
    #     ret.append((reactant, sum(ss), list(ts)[k], list(ids)[k]))
    # reactants, scores, templates, templates_idx = zip(*sorted(ret, key=lambda item : item[1], reverse=True))
        ret.append((reactant, sum(ss), sum(srs), list(ts)[0], list(ids)[0]))
    reactants, scores, scores_reference, templates, templates_idx = zip(*sorted(ret, key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(scores_reference), list(templates), list(templates_idx)



class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x

def load_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in enumerate(f):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    
    state_dict = torch.load(state_path, map_location='cpu')
    # Remove the 'module.' prefix if present (i.e., if it was saved using DataParallel)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule, template_rules
    
class MLPModel:
    def __init__(self, state_path, template_path, device='cuda', fp_dim=2048, realistic_filter=False):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules, self.rules2idx = load_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        self.net.to(device)

        self.realistic_filter = realistic_filter
        
        self.reference_net, _, _ = load_model(state_path, template_path, fp_dim)
        self.reference_net.eval()
        self.reference_net.to(device)

    def forward_topk(self, arr, topk=10):
        preds = self.net(arr)
        preds_reference = self.reference_net(arr).detach()

        if self.realistic_filter:
            preds_reference, idx_topk = torch.topk(preds_reference, k=topk)
            preds = preds.gather(1, idx_topk)
        else:
            preds, idx_topk = torch.topk(preds, k=topk)
            preds_reference = preds_reference.gather(1, idx_topk)

        return preds, preds_reference, idx_topk
    
    def run(self, mol_node, topk=10, backward=True, sample_mode=None, test=False):
        x = mol_node.mol if hasattr(mol_node, 'mol') else mol_node
        arr = smiles2array(x, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(self.device)
        # preds = self.net(arr)
        # preds = F.softmax(preds,dim=1)
        # if self.device >= 0:
        #     preds = preds.cpu()
        # probs, idx = torch.topk(preds,k=topk)
        preds, preds_reference, idx = self.forward_topk(arr, topk=topk)
        probs = F.softmax(preds, dim=1)
        probs_reference = F.softmax(preds_reference, dim=1)
        preds, preds_reference, idx, probs, probs_reference = preds.cpu(), preds_reference.cpu(), idx.cpu(), probs.cpu(), probs_reference.cpu()
        # probs = F.softmax(preds, dim=1)
        # probs_reference = F.softmax(preds_reference, dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        scores_reference = []
        templates = []
        templates_idx = []

        if sample_mode == 'template':
            # probs = probs / probs.sum()
            result = {'reactants':[],
                      'reactant_lists': [],
                      'template' : [],
                      'templates_idx': []}
            ancestors = mol_node.get_ancestors()
            for i in range(topk):
                try:
                    is_invalid = True # if current template, i.e., idx[i], is valid
                    if test:
                        template_idx = torch.argmax(probs[0]).item()
                    else:
                        template_idx = torch.multinomial(probs[0], 1).item() # TODO: sum of probabilities <= 0
                    rule = rule_k[template_idx]
                    out1 = rdchiralRunText(rule, x)
                    out1 = sorted(out1)
                    # check repeated molecules
                    for reactants in out1:
                        reactant_list = list(set(reactants.split('.')))
                        is_repeated = False
                        # for reactant in reactant_list:
                        #     if reactant in ancestors:
                        #         is_repeated = True
                        #         break
                        if not is_repeated:
                            is_invalid = False
                            result['reactants'].append(reactants)
                            result['reactant_lists'].append(reactant_list)
                            result['template'].append(rule)
                            result['templates_idx'].append(template_idx if self.realistic_filter else idx[0][template_idx].item())
                except (ValueError, RuntimeError) as e:
                    """
                    RuntimeError: Pre-condition Violation
                    Stereo atoms should be specified before specifying CIS/TRANS bond stereochemistry
                    Violation occurred on line 288 in file Code/GraphMol/Bond.h
                    Failed Expression: what <= STEREOE || getStereoAtoms().size() == 2
                    RDKIT: 2020.09.1
                    BOOST: 1_73
                    """
                    pass
                except (IndexError, KeyError) as e:
                    """
                    rdchiral bug during function call rdchiralRunText(rule, mol)
                    This error can be reprobuced by the following code:
                    mol = 'C[C@H](OC(=O)C=O)C(=O)O'
                    rule = '([#8:1]-[C:2](=[O;D1;H0:3])-[CH;D2;+0:4]=[O;H0;D1;+0:5])>>[#8:1]-[C:2](=[O;D1;H0:3])-[C@@H;D3;+0:4](-[OH;D1;+0:5])-[C@H;D3;+0:4](-[OH;D1;+0:5])-[C:2](-[#8:1])=[O;D1;H0:3]'
                    out1 = rdchiralRunText(rule, mol)
                    """
                    pass
                if is_invalid: # invalid molecule
                    result['reactants'].append('Invalid')
                    result['reactant_lists'].append(['Invalid'])
                    result['template'].append(rule)
                    result['templates_idx'].append(template_idx if self.realistic_filter else idx[0][template_idx].item())
                    probs[0][template_idx] = 0
                else:
                    return result

            return None
            
        for i , rule in enumerate(rule_k):
            out1 = []
            try:
                if backward:
                    out1 = rdchiralRunText(rule, x)
                else:
                    rxn_prod, rxn_agent, rxn_react = rule.split(">")
                    reversed_rule = '(' + rxn_react + ')>' + rxn_agent + '>' + rxn_prod[1:-1]
                    out1 = rdchiralRunText(reversed_rule, x)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0: continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    scores_reference.append(probs_reference[0][i].item()/len(out1))
                    templates.append(rule)
                    templates_idx.append(i if self.realistic_filter else idx[0][i].item())
            # out1 = rdchiralRunText(x, rule)
            except (ValueError, RuntimeError) as e:
                """
                RuntimeError: Pre-condition Violation
                Stereo atoms should be specified before specifying CIS/TRANS bond stereochemistry
                Violation occurred on line 288 in file Code/GraphMol/Bond.h
                Failed Expression: what <= STEREOE || getStereoAtoms().size() == 2
                RDKIT: 2020.09.1
                BOOST: 1_73
                """
                pass
            except (IndexError, KeyError) as e:
                """
                rdchiral bug during function call rdchiralRunText(rule, mol)
                This error can be reprobuced by the following code:
                mol = 'C[C@H](OC(=O)C=O)C(=O)O'
                rule = '([#8:1]-[C:2](=[O;D1;H0:3])-[CH;D2;+0:4]=[O;H0;D1;+0:5])>>[#8:1]-[C:2](=[O;D1;H0:3])-[C@@H;D3;+0:4](-[OH;D1;+0:5])-[C@H;D3;+0:4](-[OH;D1;+0:5])-[C:2](-[#8:1])=[O;D1;H0:3]'
                out1 = rdchiralRunText(rule, mol)
                """
                pass
        if len(reactants) == 0: return None
        reactants_d = defaultdict(list)
        for r, s, sr, t, id in zip(reactants, scores, scores_reference, templates, templates_idx):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, sr, t, id))
            else:
                reactants_d[r].append((s, sr, t, id))

        reactants, scores, scores_reference, templates, templates_idx = merge(reactants_d)
        total = sum(scores)
        total_reference = sum(scores_reference)
        scores = [s / total for s in scores]
        scores_reference = [s / total_reference for s in scores_reference]

        if sample_mode == 'molecule':
            reactant_idx = np.random.choice(len(scores), p=scores)
            return {'reactants':reactants[reactant_idx:reactant_idx+1],
                    'scores' : scores[reactant_idx:reactant_idx+1],
                    'template' : templates[reactant_idx:reactant_idx+1]}

        return {'reactants':reactants,
                'scores' : scores,
                'scores_reference' : scores_reference,
                'template' : templates,
                'templates_idx': templates_idx}

class ActionRepresentation(nn.Module):
    def __init__(self, state_dim, ac_dim, device):
        super(ActionRepresentation, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, ac_dim))
        self.model.to(device=device)
        
    def forward(self, rxn):
        return self.model(rxn)
        
        
class QValueMLP(nn.Module):
  def __init__(self, n_layers, fp_dim, rxn_dim, latent_dim, dropout_rate, device):
      super(ValueMLP, self).__init__()
      self.n_layers = n_layers
      self.rxn_dim = rxn_dim
      self.fp_dim = fp_dim
      self.latent_dim = latent_dim
      self.dropout_rate = dropout_rate
      self.device = device

      layers = []
      layers.append(nn.Linear(fp_dim+rxn_dim, latent_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(self.dropout_rate))
      for _ in range(self.n_layers - 1):
          layers.append(nn.Linear(latent_dim, latent_dim))
          layers.append(nn.ReLU())
          layers.append(nn.Dropout(self.dropout_rate))
      layers.append(nn.Linear(latent_dim, 1))

      self.layers = nn.Sequential(*layers).to(device=device)

  def forward(self, fps, rxn):
      x = torch.cat([fps, rxn], dim=1)
      x = self.layers(x)
      x = torch.log(1 + torch.exp(x))
      return x
      
         
class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate, device):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers).to(device=device)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x
        
        
class PriorMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate, device):
        super(PriorMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers).to(device=device)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        # x = torch.log(1 + torch.exp(x))
        x = torch.sigmoid(x)

        return x
        
        
        
        
        
        


class DMPNNValue(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(DMPNNValue, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.device = device

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                  bond_fdim=config['n_bond_feat'],
                                  hidden_size=config['mpn_size'],
                                  depth=config['depth'],
                                  dropout=config['dropout_mpn'],
                                  atom_message=config['atom_message'])

        self.W_vv = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)

        if config['use_attn']:
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])

        self.atom_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))
        self.bond_linear = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))

        self.graph_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")
            
            
    def forward(self, prod_tensors, prod_scopes):
        
        prod_tensors = self.to_device(prod_tensors)
        atom_scope, bond_scope = prod_scopes
        
        a_feats = self.encoder(prod_tensors, mask=None)
        atom_feats = F.relu(self.W_vc(a_feats))
        graph_vecs = torch.stack(
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])
        values = self.graph_linear(graph_vecs)
        return values

    def forward_with_smiles(self, smiles_list):
        product_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        for mol in product_mols:
            Chem.Kekulize(mol)
        
        prod_graphs = [
            MolGraph(mol=Chem.Mol(mol), rxn_class=None, use_rxn_class=False)
            for mol in product_mols
        ]
        prod_tensors, prod_scopes = get_batch_graphs(prod_graphs, use_rxn_class=False)
        values = self.forward(prod_tensors, prod_scopes)
        return values

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab

        return saveables