import torch
import numpy as np

from bgflow.utils import (
    IndexBatchIterator,
)
from bgflow import (
    DiffEqFlow,
    MeanFreeNormalDistribution,
)
from tbg.models2 import EGNN_dynamics_transferable_MD
from bgflow import BlackBoxDynamics, BruteForceEstimator
import os
import tqdm
import mdtraj as md
from torch.utils.tensorboard import SummaryWriter


data_path = "data/2AA-1-large"
n_dimensions = 3

directory = os.fsencode(data_path + "/train")
training_peptides = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        training_peptides.append(filename[:2])

max_atom_number = 0
atom_dict = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
scaling = 30

priors = {}
topologies = {}
atom_types_dict = {}
h_dict = {}
n_encodings = 13
for peptide in tqdm.tqdm(training_peptides):

    topologies[peptide] = md.load_topology(
        data_path + f"/train/{peptide}-traj-state0.pdb"
    )
    atom_types = []
    n_atoms = len(list(topologies[peptide].atoms))
    for atom_name in topologies[peptide].atoms:
        atom_types.append(atom_name.name[0])
    atom_types_dict[peptide] = np.array(
        [atom_dict[atom_type] for atom_type in atom_types]
    )
    backbone_idxs = topologies[peptide].select("backbone")
    atom_types_dict[peptide][backbone_idxs] = [5, 6, 7, 8, 9, 10, 11, 12]
    h_dict[peptide] = torch.nn.functional.one_hot(
        torch.tensor(atom_types_dict[peptide]), num_classes=n_encodings
    )
    priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()

directory = os.fsencode(data_path + "/val")
validation_peptides = []
val_priors = {}
val_topologies = {}
val_atom_types_dict = {}
val_h_dict = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdb"):
        validation_peptides.append(filename[:2])
for peptide in tqdm.tqdm(validation_peptides):

    val_topologies[peptide] = md.load_topology(
        data_path + f"/val/{peptide}-traj-state0.pdb"
    )
    atom_types = []
    n_atoms = len(list(val_topologies[peptide].atoms))
    for atom_name in val_topologies[peptide].atoms:
        atom_types.append(atom_name.name[0])
    val_atom_types_dict[peptide] = np.array(
        [atom_dict[atom_type] for atom_type in atom_types]
    )
    backbone_idxs = val_topologies[peptide].select("backbone")
    val_atom_types_dict[peptide][backbone_idxs] = [5, 6, 7, 8, 9, 10, 11, 12]
    val_h_dict[peptide] = torch.nn.functional.one_hot(
        torch.tensor(val_atom_types_dict[peptide]), num_classes=n_encodings
    )
    val_priors[peptide] = MeanFreeNormalDistribution(
        n_atoms * n_dimensions, n_atoms, two_event_dims=False
    ).cuda()
max_atom_number = 51


data = np.load(data_path + "/all_train.npy", allow_pickle=True).item()
data_val = np.load(data_path + "/all_val.npy", allow_pickle=True).item()
n_data = len(data[training_peptides[0]])
n_random = n_data // 10


net_dynamics = EGNN_dynamics_transferable_MD(
    n_particles=max_atom_number,
    h_size=n_encodings,
    device="cuda",
    n_dimension=n_dimensions,
    hidden_nf=128,
    act_fn=torch.nn.SiLU(),
    n_layers=9,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)

bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=BruteForceEstimator()
)

flow = DiffEqFlow(dynamics=bb_dynamics)

n_batch = 3
n_batch_val = 20


def resample_noise(
    peptides=training_peptides,
    priors=priors,
    h_dict=h_dict,
    n_samples=n_random,
    n_batch=n_batch,
):
    x0_list = []
    noise_list = []
    for peptide in peptides:
        n_particles = h_dict[peptide].shape[0]

        x0 = (
            priors[peptide]
            .sample(n_batch * n_samples)
            .cpu()
            .reshape(n_batch, n_samples, -1)
        )
        x0 = torch.nn.functional.pad(x0, (0, (max_atom_number - n_particles) * 3))

        noise = (
            priors[peptide]
            .sample(n_batch * n_samples)
            .cpu()
            .reshape(n_batch, n_samples, -1)
        )
        noise = torch.nn.functional.pad(noise, (0, (max_atom_number - n_particles) * 3))

        x0_list.append(x0)
        noise_list.append(noise)
    x0_list = torch.cat(x0_list)
    noise_list = torch.cat(noise_list)
    return x0_list, noise_list


val_x1_list = []
val_node_mask_batch = []
val_edge_mask_batch = []
val_h_batch = []
for peptide in validation_peptides:
    n_particles = val_h_dict[peptide].shape[0]

    x1 = torch.from_numpy(data_val[peptide])
    x1 = torch.nn.functional.pad(x1, (0, (max_atom_number - n_particles) * 3))
    val_x1_list.append(x1)
    # create the masks here as well!
    mask = torch.ones((n_batch_val, n_particles))
    # node mask as bool ornot???
    mask = torch.nn.functional.pad(
        mask, (0, (max_atom_number - n_particles))
    )  # .bool()
    edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    node_mask = mask
    edge_mask = edge_mask.reshape(-1, max_atom_number**2)
    h = (
        torch.cat(
            [
                val_h_dict[peptide],
                torch.zeros(max_atom_number - n_particles, n_encodings),
            ]
        )
        .unsqueeze(0)
        .repeat(n_batch_val, 1, 1)
    )
    val_node_mask_batch.append(node_mask)
    val_edge_mask_batch.append(edge_mask)
    val_h_batch.append(h)
val_x1_list = torch.stack(val_x1_list)
val_node_mask_batch = torch.cat(val_node_mask_batch, dim=0).cuda()
val_edge_mask_batch = torch.cat(val_edge_mask_batch, dim=0).cuda()
val_h_batch = torch.cat(val_h_batch, dim=0).cuda()


x1_list = []
node_mask_batch = []
edge_mask_batch = []
h_batch = []
for peptide in training_peptides:
    n_particles = h_dict[peptide].shape[0]

    x1 = torch.from_numpy(data[peptide])
    x1 = torch.nn.functional.pad(x1, (0, (max_atom_number - n_particles) * 3))
    x1_list.append(x1)
    # create the masks here as well!
    mask = torch.ones((n_batch, n_particles))
    # node mask as bool ornot???
    mask = torch.nn.functional.pad(
        mask, (0, (max_atom_number - n_particles))
    )  # .bool()
    edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    node_mask = mask
    edge_mask = edge_mask.reshape(-1, max_atom_number**2)
    h = (
        torch.cat(
            [h_dict[peptide], torch.zeros(max_atom_number - n_particles, n_encodings)]
        )
        .unsqueeze(0)
        .repeat(n_batch, 1, 1)
    )
    node_mask_batch.append(node_mask)
    edge_mask_batch.append(edge_mask)
    h_batch.append(h)
x1_list = torch.stack(x1_list)
node_mask_batch = torch.cat(node_mask_batch, dim=0).cuda()
edge_mask_batch = torch.cat(edge_mask_batch, dim=0).cuda()
h_batch = torch.cat(h_batch, dim=0).cuda()

batch_iter = IndexBatchIterator(n_data, n_batch)
val_batch_iter = IndexBatchIterator(n_data, n_batch_val)

optim = torch.optim.Adam(flow.parameters(), lr=5e-4)

n_epochs = 12


PATH_last = "models/tbg_backbone"
writer = SummaryWriter("logs/" + PATH_last)
try:
    checkpoint = torch.load(PATH_last)
    flow.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    loaded_epoch = checkpoint["epoch"]
    global_it = checkpoint["global_it"]
    print("Successfully loaded model")
except:
    print("Generated new model")
    loaded_epoch = 0
    global_it = 0


sigma = 0.01
for epoch in range(loaded_epoch, n_epochs):
    if epoch == 4:
        for g in optim.param_groups:
            g["lr"] = 5e-5
    if epoch == 8:
        for g in optim.param_groups:
            g["lr"] = 5e-6
    random_start_idx = torch.randint(0, n_data, (len(x1_list),)).unsqueeze(1)
    for it, idxs in enumerate(batch_iter):
        if len(idxs) != n_batch:
            continue
        peptide_idxs = torch.arange(0, len(x1_list)).repeat_interleave(len(idxs))
        it_idxs = it % n_random
        if it_idxs == 0:
            x0_list, noise_list = resample_noise()
            print(epoch, it)
            torch.save(
                {
                    "model_state_dict": flow.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "epoch": epoch,
                    "global_it": global_it,
                },
                PATH_last,
            )
        optim.zero_grad()
        x1 = x1_list[
            peptide_idxs, ((random_start_idx + idxs) % n_data).flatten()
        ].cuda()

        batchsize = x1.shape[0]
        t = torch.rand(len(x1), 1).to(x1)

        x0 = x0_list[:, it_idxs].to(x1)
        noise = noise_list[:, it_idxs].to(x1)

        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = sigma
        x = mu_t + sigma_t * noise
        ut = x1 - x0
        vt = flow._dynamics._dynamics._dynamics_function(
            t, x, h_batch, node_mask_batch, edge_mask_batch
        )
        # loss = torch.mean((vt - ut_batch) ** 2)
        # use the weighted loss instead
        loss = (
            torch.sum((vt - ut) ** 2, dim=-1)
            / node_mask_batch.int().sum(-1)
            / n_dimensions
        )
        loss = loss.mean()
        loss.backward()
        optim.step()
        writer.add_scalar("Loss/Train", loss, global_step=global_it)
        global_it += 1
    print("Validating")
    with torch.no_grad():
        loss_acum = 0
        random_start_idx = torch.randint(0, n_data, (len(val_x1_list),)).unsqueeze(1)
        for it, idxs in enumerate(val_batch_iter):
            if it == 100:
                break
            peptide_idxs = torch.arange(0, len(val_x1_list)).repeat_interleave(
                len(idxs)
            )
            x0_list, noise_list = resample_noise(
                validation_peptides, val_priors, val_h_dict, n_batch=n_batch_val
            )
            # print(val_x1_list.shape, peptide_idxs.shape, idxs.shape)
            x1 = val_x1_list[
                peptide_idxs, ((random_start_idx + idxs) % n_data).flatten()
            ].cuda()
            batchsize = x1.shape[0]
            t = torch.rand(len(x1), 1).to(x1)

            x0 = x0_list[:, it].to(x1)
            noise = noise_list[:, it].to(x1)
            # print(x1.shape, x0.shape)

            mu_t = x0 * (1 - t) + x1 * t
            sigma_t = sigma
            x = mu_t + sigma_t * noise
            ut = x1 - x0
            vt = flow._dynamics._dynamics._dynamics_function(
                t, x, val_h_batch, val_node_mask_batch, val_edge_mask_batch
            )
            # loss = torch.mean((vt - ut_batch) ** 2)
            # use the weighted loss instead
            loss = (
                torch.sum((vt - ut) ** 2, dim=-1)
                / val_node_mask_batch.int().sum(-1)
                / n_dimensions
            )
            loss_acum += loss.mean()
    writer.add_scalar("Loss/Val", loss_acum / 100, global_step=global_it)
