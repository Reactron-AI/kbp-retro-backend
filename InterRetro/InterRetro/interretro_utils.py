import os
import torch
import random
import pickle
import platform
import pandas as pd
import numpy as np

from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

def minicut(input, percentage):
    output = input[:int(len(input) * percentage / 100)]
    return output
    
def prepare_starting_molecules(filename):
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)
    return starting_mols

def smiles2array(X, fp_dim):
    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim),useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return arr

def set_seeds(seed):
    if seed < 0:
        seed = random.randint(0, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed

def get_device(device):
    if platform.system() == "Darwin":  # macOS
        return "cpu"
    return device if torch.cuda.is_available() else "cpu"

def create_exp(seed, group):
    base_dir = 'experiments'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    current_time = datetime.now().strftime("%m%d-%H:%M:%S")
    run_dir = os.path.join(base_dir, f"{group}{current_time}-s{seed}")
    os.makedirs(run_dir)
    return run_dir
