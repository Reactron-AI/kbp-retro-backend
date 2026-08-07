import importlib
import logging
import os
import sys
from argparse import Namespace

import numpy as np
import torch


def default_interretro_root():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            '..',
            '..',
            'InterRetro',
            'InterRetro'
        )
    )


def default_localretro_root():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            '..',
            '..',
            'LocalRetro'
        )
    )


def default_interretro_checkpoint(interretro_root):
    return os.path.join(
        interretro_root,
        'experiments',
        'default0429-07:53:16-s51',
        'models_epoch4_it0.pth'
    )


def _ensure_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError('%s not found: %s' % (description, path))


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class InterRetroAdapter(object):
    def __init__(self,
                 interretro_root=None,
                 checkpoint_path=None,
                 localretro_root=None,
                 localretro_model_path=None,
                 localretro_config_path=None,
                 localretro_data_dir=None,
                 topk=10,
                 device='cpu',
                 lr=1e-3,
                 adv_coef=10.0,
                 value_fp_dim=2048,
                 value_n_layers=1,
                 value_latent_dim=128,
                 value_dropout=0.1):
        self.interretro_root = os.path.abspath(
            interretro_root if interretro_root is not None else default_interretro_root()
        )
        self.localretro_root = os.path.abspath(
            localretro_root if localretro_root is not None else default_localretro_root()
        )
        self.checkpoint_path = checkpoint_path or default_interretro_checkpoint(
            self.interretro_root
        )
        self.device = torch.device(device)
        self.topk = topk

        _ensure_exists(self.interretro_root, 'InterRetro root')
        _ensure_exists(self.localretro_root, 'LocalRetro root')
        _ensure_exists(self.checkpoint_path, 'InterRetro checkpoint')

        if self.interretro_root not in sys.path:
            sys.path.insert(0, self.interretro_root)

        try:
            agents_module = importlib.import_module('agents')
        except ImportError as exc:
            raise ImportError(
                'Failed to import InterRetro agents from %s. Make sure the '
                'sibling InterRetro checkout and its dependencies are available.'
                % self.interretro_root
            ) from exc

        args = Namespace(
            agent='localretro',
            localretro_root=self.localretro_root,
            localretro_model_path=localretro_model_path,
            localretro_config_path=localretro_config_path,
            localretro_data_dir=localretro_data_dir,
            localretro_topk=topk,
            localretro_policy_batch_size=1,
            device=self.device,
            lr=lr,
            adv_coef=adv_coef,
            value_fp_dim=value_fp_dim,
            value_n_layers=value_n_layers,
            value_latent_dim=value_latent_dim,
            value_dropout=value_dropout
        )

        logging.info('Loading InterRetro from %s', self.interretro_root)
        logging.info('InterRetro checkpoint: %s', self.checkpoint_path)
        self.agent = agents_module.return_agent(args)

        state_dict = _load_checkpoint(self.checkpoint_path, self.device)
        missing_keys = sorted({'policy', 'value_fn'} - set(state_dict.keys()))
        if missing_keys:
            raise KeyError(
                'InterRetro checkpoint is missing required keys: %s' %
                ', '.join(missing_keys)
            )
        self.agent.load_state_dict(state_dict)
        for model in self.agent.nn_models.values():
            model.eval()

    def run(self, smiles, topk=10):
        result = self.agent.run(smiles, topk=topk)
        return self._format_result(result)

    def run_rank_slice(self, smiles, start_rank=0, limit=1):
        if not hasattr(self.agent, 'run_rank_slice'):
            return self.run(smiles, topk=start_rank + limit)

        result = self.agent.run_rank_slice(
            smiles,
            start_rank=start_rank,
            limit=limit
        )
        return self._format_result(result)

    def _format_result(self, result):
        if result is None or len(result.get('scores', [])) == 0:
            if result is None:
                return None
            return {
                'reactants': [],
                'scores': [],
                'template': [],
                **{
                    key: result[key]
                    for key in (
                        'root_action_rank',
                        'root_rank_start',
                        'root_rank_exhausted',
                        'next_root_rank_start'
                    )
                    if key in result
                }
            }

        scores = np.array(result['scores'], dtype=float)
        score_sum = scores.sum()
        if score_sum <= 0:
            scores = np.ones(len(scores), dtype=float) / len(scores)
        else:
            scores = scores / score_sum

        formatted = {
            'reactants': result['reactants'],
            'scores': scores.tolist(),
            'template': result.get('template', [None] * len(scores))
        }
        formatted.update({
            key: result[key]
            for key in (
                'root_action_rank',
                'root_rank_start',
                'root_rank_exhausted',
                'next_root_rank_start'
            )
            if key in result
        })
        return formatted

    def value(self, smiles):
        with torch.no_grad():
            value = self.agent._value_for_smiles([smiles]).detach().cpu().item()
        return float(value)
