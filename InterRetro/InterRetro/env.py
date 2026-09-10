from rdkit import Chem
from anytree import AnyNode, RenderTree, PreOrderIter

class RetroEnv:
    def __init__(self, building_blocks):
        self.building_blocks = building_blocks
        self.max_steps = 15
        
        self.REWARD = 1
        self.NOT_REWARD = 0
        self.DONE = 1
        self.NOT_DONE = 0

    def reset(self, mol):
        self.tobevisited_nodes = []

        unmapped_obs = self.canonicalize(mol)
        self.obs = unmapped_obs
        
        obs_mol = Chem.MolFromSmiles(unmapped_obs)
        for i, atom in enumerate(obs_mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        mapped_smi = Chem.MolToSmiles(obs_mol)

        inbb_flag = int(unmapped_obs in self.building_blocks)
        self.obs_node = AnyNode(
            id=unmapped_obs,
            action=None,
            inbb=inbb_flag,
            mapping=mapped_smi
        )
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, reactants, edits):
        # If reactants is empty => no expansion -> typically fail & done
        if len(reactants) == 0:
            return None, self.NOT_REWARD, self.DONE, None

        self.obs_node.action = edits
        for r in reactants:
            unmapped_r = self.canonicalize(r)
            inbb_flag = int(unmapped_r in self.building_blocks)
            r_node = AnyNode(
                id=unmapped_r,
                parent=self.obs_node,
                action=None,
                inbb=inbb_flag,
                mapping=r
            )
            if not inbb_flag:
                self.tobevisited_nodes.append(r_node)

        if len(self.tobevisited_nodes) > 0:
            next_node = self.tobevisited_nodes.pop()
            self.obs = next_node.id
            self.obs_node = next_node
            return self.obs, self.NOT_REWARD, self.NOT_DONE, None
        else:
            # No more nodes to expand => check if everything is in BB
            all_leaves_inbb = all(leaf.inbb == 1 for leaf in self.tree_mdp.leaves)
            reward = self.REWARD if all_leaves_inbb else self.NOT_REWARD
            self.obs = None
            self.obs_node = None
            return None, reward, self.DONE, None

    def canonicalize(self, smi):
        """
        Canonicalize a SMILES string to a standard form for comparison.
        """
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi
            mol = Chem.RemoveHs(mol)
            # Remove any old atom mapping numbers
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            return Chem.MolToSmiles(mol)
        except:
            # If there's a parse error, just return the original
            print('Warning: could not parse SMILES:', smi, flush=True)
            return smi

    def build_tree(self, agent, mol, test=False):
        obs = self.reset(mol)
        t = 0
        reward = 0
        for t in range(1, self.max_steps+1):
            if test:
                reactants, edits = agent.get_reactants(obs, sampling=False, max_steps=9)
            else:
                reactants, edits = agent.get_reactants(obs, sampling=True)

            next_obs, reward, done, info = self.step(reactants, edits)
            if done:
                break
            obs = next_obs

        # Identify the largest successful subtrees
        successful_subtrees = []
        for node in PreOrderIter(self.tree_mdp):
            if self.is_successful_subtree(node):
                if node.is_root or not self.is_successful_subtree(node.parent):
                    if node.inbb == 0:
                        successful_subtrees.append(node)
        
        depth = max(n.depth for n in PreOrderIter(self.tree_mdp))
        
        return (
            self.tree_mdp,
            successful_subtrees,
            {
                'success': int(reward == 1),
                'model_calls': t,
                'depth': depth
            }
        )

    def is_successful_subtree(self, node):
        return all(leaf.inbb == 1 for leaf in node.leaves)


    def print_tree(self, root=None):
        if root is None:
            root = self.tree_mdp
        for pre, fill, node in RenderTree(root):
            suffix = " --BB" if node.inbb == 1 else ""
            print(f"{pre}{node.id}{suffix}")