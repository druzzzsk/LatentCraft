import numpy as np
import torch
import cma


def cma_es_optimization(vae, seed_smiles, charset, prop_fn,
                        sigma0=0.5, n_iter=50, popsize=10, device="cpu"):
    """
    CMA-ES оптимизация в латентном пространстве для каждой seed-молекулы.

    Для каждого seed: encode -> CMA-ES loop (fitness = prop_fn декодированной молекулы) -> best z -> decode.

    Returns:
        optimized_smiles: list[str]
        trajectory: list[(step, mean_best_value)]
    """
    vae.eval()
    optimized_smiles = []
    all_best_values = []

    opts = cma.CMAOptions()
    opts["maxiter"] = n_iter
    opts["popsize"] = popsize
    opts["verbose"] = -9  # подавляем вывод CMA-ES

    for seed_idx, seed in enumerate(seed_smiles):
        z_seed = vae.encode_smiles([seed], charset, device=device)  # (1, hidden_dim)
        x0 = z_seed.squeeze(0).cpu().numpy()

        # Кэш для невалидных молекул — штраф как текущий минимум fitness
        # (CMA-ES минимизирует, поэтому fitness = -property)
        worst_seen = [0.0]

        def fitness(x):
            z = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            smiles = vae.decode_to_smiles(z, charset)[0]
            val = prop_fn(smiles)
            if val is None:
                return worst_seen[0]
            f = -float(val)
            if f > worst_seen[0]:
                worst_seen[0] = f
            return f

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        best_val = -fitness(x0)  # начальное значение
        best_x = x0.copy()
        best_values = [best_val]

        step = 0
        while not es.stop() and step < n_iter:
            solutions = es.ask()
            fitnesses = [fitness(x) for x in solutions]
            es.tell(solutions, fitnesses)

            current_best_f = min(fitnesses)
            current_best_val = -current_best_f

            if current_best_val > best_val:
                best_val = current_best_val
                best_x = solutions[np.argmin(fitnesses)].copy()

            best_values.append(best_val)
            step += 1

        z_best = torch.tensor(best_x, dtype=torch.float32).unsqueeze(0).to(device)
        best_smiles = vae.decode_to_smiles(z_best, charset)[0]
        optimized_smiles.append(best_smiles)
        all_best_values.append(best_values)

        if (seed_idx + 1) % 10 == 0 or seed_idx == 0:
            print(f"  CMA-ES seed {seed_idx + 1}/{len(seed_smiles)}, best={best_val:.4f}")

    # Trajectory: средний best value по всем seeds на каждом шаге
    min_len = min(len(bv) for bv in all_best_values)
    trajectory = []
    for step in range(min_len):
        mean_val = np.mean([bv[step] for bv in all_best_values])
        trajectory.append((step, float(mean_val)))

    return optimized_smiles, trajectory
