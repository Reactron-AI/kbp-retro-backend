
import os
import torch.multiprocessing as mp
from multiprocessing import Manager

from runners.serial_runner import SerialRunner

mp.set_start_method('spawn', force=True)

def worker(queue, remote, args, i):
    gpu = i % args.n_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    local_runner = SerialRunner(args)
    while True:
        cmd = remote.recv()
        if type(cmd) != str: 
            assert len(cmd) == 2  # for load model mode.
            cmd, data = cmd[0], cmd[1]
        if cmd == "load":
            local_runner.synchronise_models(*data)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd in ['run', 'test']:
            results = []
            while True:
                element = queue.get()
                if element == 'DONE': # stop element
                    break
                target_mols = [element]
                if cmd == 'run':
                    tree_list, succ_subtrees_list, info_list = local_runner.run(target_mols, test=False)
                else:
                    tree_list, succ_subtrees_list, info_list = local_runner.run(target_mols, test=True)
                results.append([tree_list, succ_subtrees_list, info_list])
            remote.send(results) # send data to continue main process. 
        else:
            raise NotImplementedError


class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.n_processes = args.n_processes
        self.parent_conns, self.worker_conns = zip(*[mp.Pipe() for _ in range(self.n_processes)])

        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.ps = [mp.Process(target=worker, args=(self.queue, self.worker_conns[i], args, i)) for i in range(self.n_processes)]
        
        for p in self.ps:
            p.daemon = True
            p.start()

    def run(self, target_mols, test):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol in target_mols:
            self.queue.put(target_mol)
        for i in range(0, self.n_processes):
            self.queue.put('DONE')

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            if test:
                self.parent_conns[i].send("test")
            else:
                self.parent_conns[i].send("run")

        # 3. prepare container to save the results.
        tree_list = []
        succ_subtrees_list = []
        info_list = []
        
        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                tree_list += result[0]
                succ_subtrees_list += result[1]
                info_list += result[2]        
                    
        return tree_list, succ_subtrees_list, info_list

    def synchronise_models(self, model_state_dict):
        cpu_state_dict = {}
        for model_name, model_params in model_state_dict.items():
            cpu_state_dict[model_name] = {k: v.cpu() for k, v in model_params.items()}
        for i in range(self.n_processes):
            self.parent_conns[i].send(("load", (cpu_state_dict, )))

    def close(self):
        for i in range(self.n_processes):
            # self.parent_conns[i].send(('close', None))
            self.parent_conns[i].send('close')