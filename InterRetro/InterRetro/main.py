import pickle
import logging
import json
import time
import wandb
import torch

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from replay_buffer import ReplayBuffer
from trajectories import TrajectoryCreator
from runners import ParallelRunner, SerialRunner
from interretro_utils import set_seeds, create_exp, minicut
from config_parser import get_args
from agents import return_agent

if __name__ == '__main__':
    args = get_args()
    args.seed = set_seeds(args.seed)
    exp_folder = create_exp(args.seed, args.group)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_folder + "/experiment_log.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Experiment run directory created in {exp_folder}")
    with open(exp_folder + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)
    if args.enable_wandb:
        wandb.init(project=args.project, config=args, group=args.group, name='{}_seed{}'.format(args.group, args.seed))
        
    routes = minicut(pickle.load(open(args.training_routes, 'rb')), args.minirun_percentage)
    logging.info(f"{len(routes)} training routes have been loaded.")
    target_mols = []
    for route in tqdm(routes):
        for rxn in route:
            try:
                smi = rxn.split('>')[0]
                target_mols.append(Chem.CanonSmiles(smi))
            except Exception:
                logging.warning(f'{smi} cannot be parsed.')
                continue
    del routes
    target_mols = list(set(target_mols))
    logging.info(f"{len(target_mols)} target molecules are loaded after removing duplicates. ")
    
    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
    logging.info(f"{len(eval_routes)} evaluation routes have been loaded.")
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    del eval_routes
    eval_mols = list(set(eval_mols))
    logging.info(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")
    
    agent = return_agent(args)
    buffer = ReplayBuffer(capacity=args.buffer_capacity, agent=agent)
    traj_creator = TrajectoryCreator(agent)

    runner = SerialRunner(args, agent=agent) if args.n_processes == 1 else ParallelRunner(args)
    runner.synchronise_models(agent.get_state_dict())
    logging.info("Training starts ...")
    for epoch in range(args.epochs):
        archive = set(target_mols)
        total_iterations = int(len(archive) / args.explore_batch_size)
        for it in tqdm(range(total_iterations), mininterval=1.0, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True):
            
            logs = {}
            
            mols = []
            for i in range(args.explore_batch_size):
                if len(archive) == 0: break
                mols.append(archive.pop())
            if len(mols) == 0: break
            
            # Explore
            start_time = time.time()
            tree_list, succ_subtrees_list, info_list = runner.run(mols, test=False)
            end_time = time.time()
            if len(info_list) == 0: continue
            logs.update({'explore/elapsed_time': end_time - start_time,
                         'explore/valid_percentage': len(tree_list) / len(mols),
                         'explore/success_rate': sum([info['success'] for info in info_list]) / len(info_list)})
            
            # Memorise  
            for tree, succ_subtrees, info in zip(tree_list, succ_subtrees_list, info_list):
                if len(succ_subtrees) > 0: # Save to buffer
                    for subtree in succ_subtrees:
                       branches = traj_creator.extract_branches(subtree)
                       for branch in branches:
                           buffer_logs = buffer.add(**branch, remember_graph=args.agent.startswith('graph2edits'))
                           logs.update(buffer_logs)
                if not info['success']: # Save to archive
                    archive.add(tree.id)


            # Training
            if getattr(agent, 'is_trainable', True) and len(buffer) > 5 * args.training_batch_size:
                for _ in range(args.update_frequency):
                    rxn_batch = buffer.sample(args.training_batch_size)
                    policy_logs = agent.fit_policy(rxn_batch)
                    value_logs = agent.fit_value(rxn_batch)
                    logs.update(policy_logs)
                    logs.update(value_logs)
            runner.synchronise_models(agent.get_state_dict())
            
            # Evaluation
            if it % args.eval_interval == 0 and args.enable_eval:
                start_time = time.time()
                tree_list, succ_subtrees_list, info_list = runner.run(eval_mols, test=True)
                end_time = time.time()
                logs.update({'evaluation/elapsed_time': end_time - start_time,
                             'evaluation/success_rate': sum([info['success'] for info in info_list]) / len(eval_mols),
                             'evaluation/depth': sum([info['depth'] for info in info_list]) / len(info_list)})

            # Save Models
            if it % args.save_interval == 0:
                torch.save(agent.get_state_dict(), f'{exp_folder}/models_epoch{epoch}_it{it}.pth')
            
            # Logs
            logs.update({'archive/size': len(archive)})
            if it != total_iterations - 1:
                logging.info(f"Training Info ep {epoch} it {it}\n" + "\n".join([f"{key}: {value:.4f}" for key, value in logs.items()]))
                if args.enable_wandb:
                    wandb.log({**logs})
        
