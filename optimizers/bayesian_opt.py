import torch
import numpy as np

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def _evaluate_smiles(vae, z, charset, prop_fn):
    """Decode z -> SMILES, compute property. Returns float or None."""
    smiles_list = vae.decode_to_smiles(z, charset)
    val = prop_fn(smiles_list[0])
    return val, smiles_list[0]


def bayesian_optimization(vae, seed_smiles, charset, prop_fn,
                          n_init=5, n_iter=20, device="cpu"):
    """
    Bayesian Optimization в латентном пространстве для каждой seed-молекулы.

    Для каждого seed: encode -> n_init случайных точек вокруг seed -> GP + qEI loop -> best z -> decode.

    Returns:
        optimized_smiles: list[str] — по одной молекуле на каждый seed
        trajectory: list[(step, mean_best_value)] — прогресс по шагам
    """
    vae.eval()
    optimized_smiles = []
    all_best_values = []

    for seed_idx, seed in enumerate(seed_smiles):
        z_seed = vae.encode_smiles([seed], charset, device=device)  # (1, hidden_dim)
        z_seed = z_seed.squeeze(0).cpu()  # (hidden_dim,)
        hidden_dim = z_seed.shape[0]

        # Инициализирующие точки: seed + случайный jitter
        init_zs = [z_seed.unsqueeze(0)]
        for _ in range(n_init - 1):
            noise = torch.randn(1, hidden_dim) * 0.3
            init_zs.append(z_seed.unsqueeze(0) + noise)
        train_X = torch.cat(init_zs, dim=0).double()  # (n_init, hidden_dim)

        # Оцениваем начальные точки
        train_Y_list = []
        for i in range(len(train_X)):
            z_i = train_X[i].unsqueeze(0).float().to(device)
            val, _ = _evaluate_smiles(vae, z_i, charset, prop_fn)
            train_Y_list.append(val if val is not None else -999.0)
        train_Y = torch.tensor(train_Y_list, dtype=torch.double).unsqueeze(-1)  # (n, 1)

        # Нормализация входного пространства вручную (чтобы не тянуть лишние зависимости)
        z_mean = train_X.mean(dim=0)
        z_std = train_X.std(dim=0).clamp(min=1e-6)

        def normalize(z):
            return (z - z_mean) / z_std

        def denormalize(z_norm):
            return z_norm * z_std + z_mean

        train_X_norm = normalize(train_X)

        best_values = [train_Y.max().item()]
        best_z = train_X[train_Y.argmax().item()].clone()

        for step in range(n_iter):
            # Нормализация Y для GP (не обязательна, но помогает)
            y_mean = train_Y.mean()
            y_std = train_Y.std().clamp(min=1e-6)
            train_Y_norm = (train_Y - y_mean) / y_std

            gp = SingleTaskGP(train_X_norm, train_Y_norm)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            best_f_norm = train_Y_norm.max()
            acq = qExpectedImprovement(model=gp, best_f=best_f_norm)

            # Оптимизируем acquisition в нормализованном пространстве
            bounds = torch.stack([
                normalize(z_seed.unsqueeze(0) - 3.0).squeeze(0),
                normalize(z_seed.unsqueeze(0) + 3.0).squeeze(0),
            ])  # (2, hidden_dim)

            candidate_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )

            z_candidate = denormalize(candidate_norm.squeeze(0)).unsqueeze(0).float().to(device)
            val, _ = _evaluate_smiles(vae, z_candidate, charset, prop_fn)
            val = val if val is not None else -999.0

            new_x_norm = candidate_norm.detach().double()
            new_y = torch.tensor([[val]], dtype=torch.double)
            new_y_norm = (new_y - y_mean) / y_std

            train_X_norm = torch.cat([train_X_norm, new_x_norm], dim=0)
            train_Y = torch.cat([train_Y, new_y], dim=0)

            if val > best_values[-1]:
                best_values.append(val)
                best_z = denormalize(candidate_norm.squeeze(0)).detach()
            else:
                best_values.append(best_values[-1])

        # Декодируем лучший z
        z_best = best_z.unsqueeze(0).float().to(device)
        best_smiles = vae.decode_to_smiles(z_best, charset)[0]
        optimized_smiles.append(best_smiles)
        all_best_values.append(best_values)

        if (seed_idx + 1) % 10 == 0 or seed_idx == 0:
            print(f"  BO seed {seed_idx + 1}/{len(seed_smiles)}, best={best_values[-1]:.4f}")

    # Trajectory: средний best value по всем seeds на каждом шаге
    trajectory = []
    n_steps = len(all_best_values[0]) if all_best_values else 0
    for step in range(n_steps):
        mean_val = np.mean([bv[step] for bv in all_best_values])
        trajectory.append((step, float(mean_val)))

    return optimized_smiles, trajectory
