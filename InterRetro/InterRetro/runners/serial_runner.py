
from interretro_utils import prepare_starting_molecules

import logging
import torch.multiprocessing as mp

from env import RetroEnv
from agents import return_agent

mp.set_start_method('spawn', force=True)

class SerialRunner:
    def __init__(self, args, agent=None):
        self.args = args
        self.building_blocks = prepare_starting_molecules(args.building_blocks_path)
        self.agent = agent if agent is not None else return_agent(args)
        self.uses_external_agent = agent is not None
        self.env = RetroEnv(building_blocks=self.building_blocks)

    
    def run(self, target_mols, test):
        tree_list = []
        succ_subtrees_list = []
        info_list = []
        for mol in target_mols:
            try:
                tree, succ_subtrees, info = self.env.build_tree(self.agent, mol, test=test)
            except Exception as e:
                logging.warning(f'{e}')
                continue
            tree_list.append(tree)
            succ_subtrees_list.append(succ_subtrees)
            info_list.append(info)
        return tree_list, succ_subtrees_list, info_list

    def close(self):
        pass

    def synchronise_models(self, model_state_dict):
        if self.uses_external_agent:
            return
        self.agent.load_state_dict(model_state_dict)
