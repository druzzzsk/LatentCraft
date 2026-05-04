import torch


def gradient_ascent(vae, predictor, seed_smiles, charset, steps=80, lr=0.1, device="cpu"):
    z = vae.encode_smiles(seed_smiles, charset, device=device)  # (n, hidden_dim)
    z = z.clone().detach()

    trajectory = []

    for step in range(steps):
        z.requires_grad_(True)
        pred = predictor(z)
        pred.sum().backward()

        with torch.no_grad():
            z = z + lr * z.grad
        z = z.detach()

        if (step + 1) % 20 == 0:
            with torch.no_grad():
                current_pred = predictor(z).mean().item()
            trajectory.append((step + 1, current_pred))

    optimized_smiles = vae.decode_to_smiles(z, charset)
    return optimized_smiles, trajectory
