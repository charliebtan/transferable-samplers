import torch
import numpy as np

from bgflow.utils import IndexBatchIterator
from bgflow import DiffEqFlow, MeanFreeNormalDistribution
from tbg.models2 import EGNN_dynamics_AD2_cat
from bgflow import BlackBoxDynamics, BruteForceEstimator

n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions


# atom types for backbone
atom_types = np.array(
    [1, 0, 0, 0, 4, 3, 5, 0, 6, 0, 1, 0, 0, 0, 7, 3, 8, 0, 1, 0, 0, 0]
)
h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))


# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator = BruteForceEstimator()
net_dynamics = EGNN_dynamics_AD2_cat(
    n_particles=n_particles,
    device="cuda",
    n_dimension=dim // n_particles,
    h_initial=h_initial,
    hidden_nf=64,
    act_fn=torch.nn.SiLU(),
    n_layers=5,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)

bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator
)

flow = DiffEqFlow(dynamics=bb_dynamics)


n_batch = 256
data_path = "data/AD2/AD2_weighted.npy"
data_smaller = torch.from_numpy(np.load(data_path)).float()
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

n_epochs = 1000

PATH_last = "models/Flow-Matching-AD2-amber-weighted-backbone"

sigma = 0.01
for epoch in range(n_epochs):
    if epoch == 500:
        for g in optim.param_groups:
            g["lr"] = 5e-5
    for it, idx in enumerate(batch_iter):
        optim.zero_grad()

        x1 = data_smaller[idx].cuda()
        batchsize = x1.shape[0]

        t = torch.rand(batchsize, 1).cuda()
        x0 = prior_cpu.sample(batchsize).cuda()

        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        noise = prior.sample(batchsize)
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optim.step()
    if epoch % 100 == 0:
        print(epoch)
        torch.save(
            {
                "model_state_dict": flow.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
            },
            PATH_last,
        )
