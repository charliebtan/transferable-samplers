import tqdm
import torch
import numpy as np
from bgflow.utils import as_numpy
from bgflow import DiffEqFlow, BoltzmannGenerator, MeanFreeNormalDistribution
from bgflow import BlackBoxDynamics, BruteForceEstimator
from tbg.models2 import EGNN_dynamics_AD2_cat


n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions

scaling = 10

# atom types for backbone
n_particles = 22
n_dimensions = 3
dim = n_particles * n_dimensions


atom_types = np.arange(22)
atom_types[[1, 2, 3]] = 2
atom_types[[19, 20, 21]] = 20
atom_types[[11, 12, 13]] = 12
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

bg = BoltzmannGenerator(prior, flow, prior).cuda()


class BruteForceEstimatorFast(torch.nn.Module):
    """
    Exact bruteforce estimation of the divergence of a dynamics function.
    """

    def __init__(self):
        super().__init__()
        self.n = 0

    def forward(self, dynamics, t, xs):
        self.n += 1

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            x = [xs[:, [i]] for i in range(xs.size(1))]

            dxs = dynamics(t, torch.cat(x, dim=1))

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = 0
            for i in range(xs.size(1)):
                divergence += torch.autograd.grad(
                    dxs[:, [i]], x[i], torch.ones_like(dxs[:, [i]]), retain_graph=True
                )[0]

        return dxs, -divergence.view(-1, 1)


brute_force_estimator_fast = BruteForceEstimatorFast()
# use OTD in the evaluation process
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = 1e-4
bg.flow._integrator_rtol = 1e-4
flow._use_checkpoints = False
flow._kwargs = {}


filename = "Flow-Matching-AD2-amber-weighted-encoding"


PATH_last = f"models/{filename}"
checkpoint = torch.load(PATH_last)
flow.load_state_dict(checkpoint["model_state_dict"])

n_samples = 400
n_sample_batches = 500
latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
dlogp_np = np.empty(shape=(0))
print(f"Start sampling with {filename}")

for i in tqdm.tqdm(range(n_sample_batches)):
    with torch.no_grad():
        samples, latent, dlogp = bg.sample(n_samples, with_latent=True, with_dlogp=True)
        print("n / i = ", brute_force_estimator_fast.n / (i+1))
        latent_np = np.append(latent_np, latent.detach().cpu().numpy())
        samples_np = np.append(samples_np, samples.detach().cpu().numpy())

        dlogp_np = np.append(dlogp_np, as_numpy(dlogp))

    latent_np = latent_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)
    np.savez(
        f"result_data/{filename}",
        latent_np=latent_np,
        samples_np=samples_np,
        dlogp_np=dlogp_np,
    )
