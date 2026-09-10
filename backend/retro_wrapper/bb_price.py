"""
Building-block price lookup ($/g), keyed by canonical SMILES.

Backed by KBP-data/bb_price_index.pkl, a precomputed {canonical_smiles: price}
dict built by KBP-data/build_bb_price_index.py from bb_curation.csv (Chemspace
SMILES + $/g, non-canonical as shipped). Canonicalization here matches
retro_star/route_utils.py::canonicalize_smiles so route node SMILES (already
canonical) hit the index directly, and arbitrary query SMILES still resolve.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

RETRO_STAR_PATH = Path(__file__).parent.parent.parent / 'retro_star'
if str(RETRO_STAR_PATH) not in sys.path:
    sys.path.insert(0, str(RETRO_STAR_PATH))

from retro_star.route_utils import canonicalize_smiles  # noqa: E402

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent.parent.parent.parent / 'KBP-data' / 'bb_price_index.pkl'

_price_index: Optional[Dict[str, float]] = None


def load_price_index(path: Path = INDEX_PATH) -> Dict[str, float]:
    global _price_index
    if _price_index is not None:
        return _price_index

    if not path.exists():
        logger.warning(f'⚠️  Building-block price index not found at {path}. '
                        f'/api/v1/bb-price will return null for every SMILES until it is built '
                        f'(run KBP-data/build_bb_price_index.py).')
        _price_index = {}
        return _price_index

    logger.info(f'Loading building-block price index from {path}...')
    with open(path, 'rb') as f:
        _price_index = pickle.load(f)
    logger.info(f'✅ Loaded {len(_price_index)} building-block prices')
    return _price_index


def lookup_prices(smiles_list: List[str]) -> Dict[str, Optional[float]]:
    """Returns {input_smiles: price_per_g_usd or None} for each input SMILES."""
    index = load_price_index()
    result = {}
    for smiles in smiles_list:
        canon = canonicalize_smiles(smiles)
        result[smiles] = index.get(canon)
    return result
