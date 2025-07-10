import math

import torch
import torch.nn as nn

from src.models.components.tbg.egnn import EGNN
from src.models.components.tbg.utils import remove_mean


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        num_particles,
        num_dimensions,
        channels=64,
        act_fn=torch.nn.SiLU(),
        num_layers=5,
        recurrent=True,
        attention=True,
        tanh=True,
        agg="sum",
        M=128,
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=2,
            in_edge_nf=1,
            hidden_nf=channels,
            act_fn=act_fn,
            n_layers=num_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self._n_particles = num_particles
        self._n_dimensions = num_dimensions
        self.edges = self._create_edges()
        self._edges_dict = {}
        # Count function calls
        self.counter = 0
        self.M = M

    def forward(self, t, x, d_base=None, *args, **kwargs):
        t = t.view(-1, 1).to(x.device)
        d_base = d_base.view(-1, 1).to(x.device) if d_base is not None else None

        if t.numel() == 1:
            t = t.repeat(x.shape[0], 1)

        if d_base is None:
            d_base = torch.ones_like(t) * math.log2(self.M)  # so it defaults to non-shortcut model
        elif d_base.numel() == 1:
            d_base = d_base.repeat(x.shape[0], 1)

        n_batch = x.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, device=x.device)
        edges = [edges[0], edges[1]]
        # Changed by Leon
        x = x.reshape(n_batch * self._n_particles, self._n_dimensions).clone()
        h = torch.ones(n_batch, self._n_particles, 2, device=x.device)

        td = torch.cat([t, d_base], dim=-1)
        h = h * td.unsqueeze(1)
        h = h.reshape(n_batch * self._n_particles, 2)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimensions)
        vel = remove_mean(vel)
        # Changed by Leon
        self.counter += 1
        return vel.view(n_batch, self._n_particles * self._n_dimensions)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
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
