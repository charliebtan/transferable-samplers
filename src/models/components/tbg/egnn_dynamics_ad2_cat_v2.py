import math

import numpy as np
import torch
import torch.nn as nn

from src.models.components.tbg.egnn import TEGNN
from src.models.components.tbg.utils import remove_mean


class EGNN_dynamics_AD2_cat_v2(nn.Module):
    def __init__(
        self,
        num_particles,
        num_dimensions,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=5,  # changed to match AD2_classical_train_tgb_full.py
        recurrent=True,
        attention=True,  # changed to match AD2_classical_train_tgb_full.py
        tanh=True,  # changed to match AD2_classical_train_tgb_full.py
        agg="sum",
        M=128,
    ):
        super().__init__()
        self._num_particles = num_particles
        self._num_dimensions = num_dimensions
        # Initial one hot encoding of the different element types
        self.h_initial = self.get_h_initial()

        h_size = self.h_initial.size(1)

        self.egnn = TEGNN(
            in_node_nf=h_size,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self.edges = self._create_edges()
        self._edges_dict = {}
        # Count function calls
        self.counter = 0
        self.h_init = None

    def get_h_initial(self):
        if self._num_particles == 22:
            atom_types = np.arange(22)
            atom_types[[1, 2, 3]] = 2
            atom_types[[19, 20, 21]] = 20
            atom_types[[11, 12, 13]] = 12
            return torch.nn.functional.one_hot(torch.tensor(atom_types))
        if self._num_particles == 33:
            atom_types = np.arange(33)
            atom_types[[1, 2, 3]] = 2
            atom_types[[9, 10, 11]] = 10
            atom_types[[19, 20, 21]] = 18
            atom_types[[29, 30, 31]] = 31
            h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
            return h_initial
        if self._num_particles == 42:
            atom_types = np.arange(42)
            atom_types[[1, 2, 3]] = 2
            atom_types[[11, 12, 13]] = 12
            atom_types[[21, 22, 23]] = 22
            atom_types[[31, 32, 33]] = 32
            atom_types[[39, 40, 41]] = 40
            h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
            return h_initial

    def forward(self, t, x, d_base=None, *args, **kwargs):
        n_batch = x.shape[0]
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.repeat(n_batch, 1)
        if t.shape != (n_batch, 1):
            t = t.repeat(n_batch)
        # t is always shape (n_batch, 1)
        t = t.flatten()
        # t = t.repeat(1, self._num_particles)
        # t = t.reshape(n_batch * self._num_particles)

        edges = self._cast_edges2batch(self.edges, n_batch, self._num_particles, device=x.device)
        edges = [edges[0], edges[1]]

        # Changed by Leon
        x = x.reshape(n_batch * self._num_particles, self._num_dimensions)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(self.h_initial.to(x), x, t, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._num_particles, self._num_dimensions)
        vel = remove_mean(vel)
        self.counter += 1
        return vel.view(n_batch, self._num_particles * self._num_dimensions)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._num_particles):
            for j in range(i + 1, self._num_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes, device):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(device)
            cols_total = torch.cat(cols_total).to(device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
