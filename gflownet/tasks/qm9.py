import tarfile
import time
import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Dict

import rdkit.Chem as Chem
from rdkit import RDLogger

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch.utils.data import Dataset

from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionType, GraphActionCategorical
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance


class QM9Dataset(Dataset):
    def __init__(self, xyz_file, train=True, split_seed=142857, ratio=0.9):
        f = tarfile.TarFile(xyz_file, 'r')
        labels = ['rA', 'rB', 'rC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        all_mols = []
        for pt in f:
            pt = f.extractfile(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
        df = pd.DataFrame(all_mols, columns=['SMILES']+labels)
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(df))
        rng.shuffle(idcs)
        if train:
            self.idcs = idcs[:int(np.floor(ratio * len(df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(df))):]
        self.df = df

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return self.df['SMILES'][self.idcs[idx]], np.exp(-self.df['gap'][self.idcs[idx]] / 0.6221)

    
class Model(nn.Module):
    def __init__(self, env_ctx, num_emb=64, initial_Z_guess=3):
        super().__init__()
        self.x2h = nn.Linear(env_ctx.num_node_dim, num_emb)
        self.e2h = nn.Linear(env_ctx.num_edge_dim, num_emb)
        self.c2h = nn.Linear(env_ctx.num_cond_dim, num_emb)
        self.graph2emb = nn.ModuleList(
            sum([[
                gnn.TransformerConv(num_emb, num_emb, edge_dim=num_emb),
                gnn.GENConv(num_emb, num_emb, num_layers=1, aggr='add'),
            ] for i in range(6)], []))

        def h2l(nl):
            return nn.Sequential(nn.Linear(num_emb, num_emb), nn.LeakyReLU(),
                                 nn.Linear(num_emb, num_emb), nn.LeakyReLU(),
                                 nn.Linear(num_emb, nl))

        self.emb2add_edge = h2l(1)
        self.emb2add_node = h2l(env_ctx.num_new_node_values)
        self.emb2set_node_attr = h2l(env_ctx.num_node_attr_logits)
        self.emb2set_edge_attr = h2l(env_ctx.num_edge_attr_logits)
        self.emb2stop = h2l(1)
        self.emb2reward = h2l(1)
        self.o2o = nn.Sequential(
            gnn.TransformerConv(num_emb, num_emb, edge_dim=None),
            gnn.TransformerConv(num_emb, num_emb, edge_dim=None))
        self.logZ = nn.Sequential(nn.Linear(env_ctx.num_cond_dim, num_emb * 2), nn.LeakyReLU(), nn.Linear(num_emb * 2, 1))
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.SetNodeAttr,
            GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr
        ]
        self._xinfo = {}

    def forward(self, g: gd.Batch, cond: torch.tensor):
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        for i, layer in enumerate(self.graph2emb):
            if isinstance(layer, nn.BatchNorm1d):
                o = layer(o)
            else:
                o = layer(o, g.edge_index, e)
            self._xinfo[f'layer {i}'] = (o.min(), o.mean(), o.max())
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat(
            [g.edge_index,
             torch.stack([u, v]),
             torch.stack([v, u])],
            1)
        # Cat the node embedding to o
        o = torch.cat([o, c], 0)
        # Do a forward pass, and remove the extraneous `c` we just concatenated
        for layer in self.o2o:
            o = o + layer(o, aug_edge_index)
        if 0:
            o = o[:-c.shape[0]]
            glob = gnn.global_mean_pool(o, g.batch)
        else:
            glob = gnn.global_mean_pool(o, torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0))
            o = o[:-c.shape[0]]
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            logits=[
                self.emb2stop(glob),
                self.emb2add_node(o),
                self.emb2set_node_attr(o),
                self.emb2add_edge(o[ne_row] + o[ne_col]),
                self.emb2set_edge_attr(o[e_row] + o[e_col]),
            ],
            keys=[None, 'x', 'x', 'non_edge_index', 'edge_index'],
            types=self.action_type_order,
        )
        for i, l in enumerate(cat.logits):
            self._xinfo[f'logit {i}'] = (l.min(), l.mean(), l.max()) if l.shape[0] > 0 else (0,0,0)
        return cat, self.emb2reward(glob)


class QM9Trial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        mp.set_start_method('spawn')
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.ctx = MolBuildingEnvContext(['H', 'C', 'N', 'F', 'O'], num_cond_dim=1)
        self.model = context.wrap_model(Model(
            self.ctx,
            num_emb=context.get_hparam('num_emb')))
        self.opt = context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), context.get_hparam('learning_rate')))
        self.tb = TrajectoryBalance(self.env, self.ctx, self.rng, random_action_prob=0.01, max_nodes=9,
                                    epsilon=context.get_hparam('tb_epsilon'))
        self.tb.reward_loss_multiplier = context.get_hparam('reward_loss_multiplier')
        self.temperature_sample_dist = context.get_hparam('temperature_sample_dist')
        self.temperature_dist_params = eval(context.get_hparam('temperature_dist_params'))

    def _sample_temperatures(self, n):
        if self.temperature_sample_dist == 'gamma':
            return self.rng.gamma(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == 'uniform':
            return self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
        raise ValueError(self.temperature_sample_dist)

    def build_training_data_loader(self) -> DataLoader:
        data = QM9Dataset(self.context.get_data_config()['path'], train=True)
        return DataLoader(data, batch_size=self.context.get_per_slot_batch_size())
    
    def build_validation_data_loader(self) -> DataLoader:
        data = QM9Dataset(self.context.get_data_config()['path'], train=False)
        return DataLoader(data, batch_size=self.context.get_per_slot_batch_size())

    def train_batch(self, batch: Tuple[List[str], torch.Tensor], epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self.model, 'device'):
            self.model.device = self.context.to_device(torch.ones(1)).device
        smiles, flat_rewards = batch
        mb_size = len(smiles)
        t = [time.time()]
        graphs = [self.ctx.mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
        temp = self._sample_temperatures(mb_size * 2)
        cond_info = self.context.to_device(torch.tensor(temp).reshape((-1, 1)))
        rewards = flat_rewards ** cond_info[:mb_size, 0]
        t += [time.time()]
        offline_losses, off_info = self.tb.compute_data_losses(
            self.env, self.ctx, self.model, graphs, rewards, cond_info=cond_info[:mb_size])
        t += [time.time()]
        online_losses = self.tb.sample_model_losses(
            self.env, self.ctx, self.model, mb_size, cond_info=cond_info[mb_size:])
        t += [time.time()]
        avg_online_loss = online_losses.mean()
        avg_offline_loss = offline_losses.mean()
        loss = (avg_offline_loss + avg_online_loss) / 2
        
        self.context.backward(loss)
        self.context.step_optimizer(
            self.opt,
            clip_grads=lambda params: torch.nn.utils.clip_grad_value_(params, 1))
        t += [time.time()]
        #print(' '.join(f"{t[i+1]-t[i]:.3f}" for i in range(len(t)-1)))
        return {'loss': loss,
                'avg_online_loss': avg_online_loss,
                'avg_offline_loss': avg_offline_loss,
                'reward_loss': off_info['reward_losses'].mean().item(),
                'unnorm_traj_losses': off_info['unnorm_traj_losses'].mean().item()}

    def evaluate_batch(self, batch: Tuple[List[str], torch.Tensor]) -> Dict[str, Any]:
        if not hasattr(self.model, 'device'):
            self.model.device = self.context.to_device(torch.ones(1)).device
        smiles, flat_rewards = batch
        mb_size = len(smiles)
        graphs = [self.ctx.mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
        temp = self._sample_temperatures(mb_size)
        cond_info = self.context.to_device(torch.tensor(temp).reshape((-1, 1)))
        rewards = flat_rewards ** cond_info[:, 0]
        losses, info = self.tb.compute_data_losses(self.env, self.ctx,
                                             self.model, graphs, rewards,
                                             cond_info=cond_info)
        return {'validation_loss': losses.mean().item(),
                'reward_loss': info['reward_losses'].mean().item(),
                'unnorm_traj_losses': info['unnorm_traj_losses'].mean().item()}

class DummyContext:

    def __init__(self, hps, device):
        self.hps = hps
        self.dev = device
    
    def wrap_model(self, model):
        return model.to(self.dev)

    def wrap_optimizer(self, opt):
        return opt

    def get_hparam(self, hp):
        return self.hps[hp]

    def get_data_config(self):
        return {'path': '/data/chem/qm9.xyz.tar'}

    def get_per_slot_batch_size(self):
        return self.hps['global_batch_size']

    def to_device(self, x):
        return x.to(self.dev)

    def backward(self, loss):
        loss.backward()

    def step_optimizer(self, opt):
        opt.step()
        opt.zero_grad()
    
def main():
    hps = {
        'learning_rate': 1e-4,
        'global_batch_size': 64,
        'num_emb': 64,
        'tb_epsilon': -60,
        'reward_loss_multiplier': 1,
        'temperature_sample_dist': 'uniform',
        'temperature_dist_params': '(0.5, 8)',
    }
    dummy_context = DummyContext(hps, torch.device('cuda'))
    trial = QM9Trial(dummy_context)

    train_dl = trial.build_training_data_loader()
    #valid_dl = trial.build_validation_data_loader()

    for epoch in range(10):
        for it, batch in enumerate(train_dl):
            batch = (batch[0], batch[1].to(dummy_context.dev))
            trial.train_batch(batch, epoch, it)
            if it >= 10:
                break
        break

if __name__ == '__main__':
    main()
