import mdtraj as md
import numpy as np

for sequence in ["GYDPETGTWG", "YQNPDGSQA"]:

    # # Load trajectory (requires both .xtc and topology file, e.g. .pdb)
    # traj = md.load_xtc(f"../scratch/unisim_pepmd_results_1000_nrg_after/{sequence}/{sequence}_model_ode50_inf100000_guidance0.05.xtc", top=f"../scratch/old_test_set/raw_data/{sequence}-traj-state0.pdb")

    # positions = traj.xyz

    # print(positions.shape)

    # np.save(f"{sequence}_unisim.npy", positions)

    not_found = 0

    for maxiter in [100, 1000]:
        all_arrays = []
        for i in range(500):
            try:
                all_arrays.append(np.load(f"../scratch/bioemu_results/{sequence}_maxiter{maxiter}/{sequence}_md_equil__parallel{i}.npy"))
            except FileNotFoundError:
                print(f"File not found for {sequence}, maxiter {maxiter}, parallel {i}")
                not_found += 1
                continue
        array = np.concatenate(all_arrays, axis=0)
        np.save(f"{sequence}_bioemu_maxiter{maxiter}.npy", array)

        print(array.shape, not_found)