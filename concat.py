import torch

path = "/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al6_cfm_v2_v21_run3/"

samples_list = []
log_p_list = []
prior_samples_list = []

for i in range(52):
    dict = torch.load(f"{path}/{i}/test_samples.pt")

    samples_list.append(dict["samples"])
    log_p_list.append(dict["log_p"])
    prior_samples_list.append(dict["prior_samples"])

samples = torch.cat(samples_list, dim=0)
log_p = torch.cat(log_p_list, dim=0)
prior_samples = torch.cat(prior_samples_list, dim=0)

torch.save({
    "samples": samples,
    "log_p": log_p,
    "prior_samples": prior_samples
}, f"{path}/test_samples.pt")

