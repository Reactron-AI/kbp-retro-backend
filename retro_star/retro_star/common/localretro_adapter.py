import importlib
import logging
import os
import sys
from collections import OrderedDict

import torch


def default_localretro_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'LocalRetro')
    )


def default_localretro_paths(localretro_root):
    return {
        'config_path': os.path.join(localretro_root, 'data', 'configs', 'default_config.json'),
        'model_path': os.path.join(localretro_root, 'models', 'LocalRetro_USPTO_50K.pth'),
        # 'model_path': os.path.join(localretro_root, 'models', 'LocalRetro_TotSyn.pth'),
        # 'model_path': os.path.join(localretro_root, 'models', 'LocalRetro_Pistachio_epoch_45.pth'),
        
        'data_dir': os.path.join(localretro_root, 'data', 'USPTO_50K')
        # 'data_dir': os.path.join(localretro_root, 'data', 'TotSyn')
        # 'data_dir': os.path.join(localretro_root, 'data', 'Pistachio')
    }


def _ensure_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError('%s not found: %s' % (description, path))


class LocalRetroAdapter(object):
    def __init__(self, localretro_root, model_path=None, config_path=None,
                 data_dir=None, device='cpu'):
        self.localretro_root = os.path.abspath(
            localretro_root if localretro_root is not None else default_localretro_root()
        )
        defaults = default_localretro_paths(self.localretro_root)
        self.model_path = model_path or defaults['model_path']
        self.config_path = config_path or defaults['config_path']
        self.data_dir = data_dir or defaults['data_dir']
        self.device = torch.device(device)

        _ensure_exists(self.localretro_root, 'LocalRetro root')
        _ensure_exists(self.model_path, 'LocalRetro model checkpoint')
        _ensure_exists(self.config_path, 'LocalRetro config')
        _ensure_exists(self.data_dir, 'LocalRetro data directory')

        if self.localretro_root not in sys.path:
            sys.path.insert(0, self.localretro_root)

        try:
            localretro_module = importlib.import_module('Retrosynthesis')
        except ImportError as exc:
            raise ImportError(
                'Failed to import LocalRetro from %s. Make sure the sibling LocalRetro '
                'checkout and its dependencies are available.' % self.localretro_root
            ) from exc

        logging.info('Loading LocalRetro from %s', self.localretro_root)
        logging.info('LocalRetro checkpoint: %s', self.model_path)
        logging.info('LocalRetro config: %s', self.config_path)
        logging.info('LocalRetro data dir: %s', self.data_dir)
        self.model = localretro_module.LocalRetro({
            'model_path': self.model_path,
            'config_path': self.config_path,
            'data_dir': self.data_dir,
            'device': self.device
        })

    def run(self, smiles, topk=10):
        results_df = self.model.retrosnythesis(smiles, top_k=topk)
        return self._format_results_df(smiles, results_df, topk=topk)

    def run_rank_slice(self, smiles, start_rank=0, limit=1):
        end_rank = start_rank + limit
        sliced_results = False
        if hasattr(self.model, 'retrosnythesis_rank_slice'):
            results_df, metadata = self.model.retrosnythesis_rank_slice(
                smiles,
                start_rank=start_rank,
                limit=limit
            )
            sliced_results = True
        else:
            results_df = self.model.retrosnythesis(smiles, top_k=end_rank)
            metadata = {
                'root_action_rank': start_rank,
                'root_rank_start': start_rank,
                'root_rank_exhausted': (
                    results_df is None or len(results_df.index) <= start_rank + 1
                ),
                'next_root_rank_start': end_rank
            }

        ranked = self._ranked_reactions(smiles, results_df, topk=None)
        metadata = {
            'root_action_rank': start_rank,
            'root_rank_start': start_rank,
            'root_rank_exhausted': metadata.get(
                'root_rank_exhausted',
                len(ranked) == 0
            ),
            'next_root_rank_start': metadata.get(
                'next_root_rank_start',
                end_rank
            )
        }
        if not sliced_results:
            ranked = ranked[start_rank:end_rank]
        if metadata['root_rank_exhausted'] or len(ranked) == 0:
            return {
                'reactants': [],
                'scores': [],
                'template': [],
                **metadata
            }
        return self._format_ranked(ranked[:limit], metadata=metadata)

    def _format_results_df(self, smiles, results_df, topk=10):
        ranked = self._ranked_reactions(smiles, results_df, topk=topk)
        if len(ranked) == 0:
            return None

        return self._format_ranked(ranked)

    def _ranked_reactions(self, smiles, results_df, topk=10):
        if results_df is None or len(results_df) == 0:
            return []

        merged = OrderedDict()
        valid_rank = 0
        for _, row in results_df.iterrows():
            reactants = row.get('SMILES')
            template = row.get('Local reaction template')
            score = row.get('Score')

            if reactants == smiles:
                continue
            if reactants is None or template is None or score is None:
                continue
            if reactants != reactants or template != template or score != score:
                continue

            reactant_parts = sorted([part for part in str(reactants).split('.') if part])
            if len(reactant_parts) == 0:
                continue

            valid_rank += 1
            reactant_key = '.'.join(reactant_parts)
            if reactant_key not in merged:
                merged[reactant_key] = {
                    'score': float(score),
                    'template': template,
                    'template_rank': valid_rank
                }
            else:
                merged[reactant_key]['score'] += float(score)
                merged[reactant_key]['template_rank'] = min(
                    merged[reactant_key]['template_rank'],
                    valid_rank
                )

        if len(merged) == 0:
            return []

        ranked = sorted(
            merged.items(),
            key=lambda item: item[1]['score'],
            reverse=True
        )
        if topk is not None:
            ranked = ranked[:topk]

        return ranked

    def _format_ranked(self, ranked, metadata=None):
        metadata = metadata or {}
        if len(ranked) == 0:
            return {
                'reactants': [],
                'scores': [],
                'template': [],
                **metadata
            }

        total_score = sum(item[1]['score'] for item in ranked)
        if total_score <= 0:
            return None

        reactants = [item[0] for item in ranked]
        scores = [item[1]['score'] / total_score for item in ranked]
        templates = [item[1]['template'] for item in ranked]
        template_ranks = [item[1].get('template_rank') for item in ranked]

        result = {
            'reactants': reactants,
            'scores': scores,
            'template': templates,
            'template_ranks': template_ranks
        }
        result.update(metadata)
        return result
