import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocessing import onehot_to_smiles
from models.smiles_vae import Decoder


class FpEncoder(nn.Module):
    """MLP encoder over concatenated Morgan fingerprint + z-score normalized descriptors."""

    def __init__(self, fp_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, hidden_dim)
        self.fc_log_var = nn.Linear(256, hidden_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mean(h), self.fc_log_var(h)


class FpVAE(nn.Module):
    """
    Asymmetric VAE: encoder reads Morgan FP + RDKit descriptors, decoder reconstructs SMILES.

    desc_mean and desc_std are registered as buffers so they are saved in the checkpoint
    and available without a separate file at inference time.
    """

    def __init__(self, fp_dim, max_len, n_chars, hidden_dim=196, gru_depth=4, gru_dim=488):
        super().__init__()
        self.fp_dim = fp_dim
        self.max_len = max_len
        self.n_chars = n_chars
        self.hidden_dim = hidden_dim

        self.encoder = FpEncoder(fp_dim, hidden_dim)
        self.decoder = Decoder(max_len, n_chars, gru_depth, gru_dim, hidden_dim)

        # Scaler stats — initialized to identity, updated after data loading
        from data.fp_features import DESC_DIM
        self.register_buffer("desc_mean", torch.zeros(DESC_DIM))
        self.register_buffer("desc_std", torch.ones(DESC_DIM))

    def set_scaler_stats(self, desc_mean, desc_std):
        """Copy scaler stats into model buffers (called once after data loading)."""
        self.desc_mean.copy_(torch.tensor(desc_mean, dtype=torch.float32))
        self.desc_std.copy_(torch.tensor(desc_std, dtype=torch.float32))

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, fp_vec, smiles_onehot):
        """
        fp_vec:       (batch, fp_dim)             — normalized FP+descriptor vector
        smiles_onehot:(batch, max_len, n_chars)   — ground truth for teacher forcing
        """
        z_mean, z_log_var = self.encoder(fp_vec)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z, smiles_onehot)
        return x_recon, z_mean, z_log_var

    def encode(self, fp_vec):
        z_mean, _ = self.encoder(fp_vec)
        return z_mean

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n, device="cpu"):
        return torch.randn(n, self.hidden_dim, device=device)

    def loss_function(self, smiles_onehot, x_recon_logits, z_mean, z_log_var, kl_weight=1.0):
        batch = smiles_onehot.shape[0]

        targets = smiles_onehot.argmax(dim=-1).reshape(-1)
        logits = x_recon_logits.reshape(-1, self.n_chars)
        recon_loss = F.cross_entropy(logits, targets, reduction="sum") / batch

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        )

        total = recon_loss + kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def encode_smiles(self, smiles_list, charset, device="cpu"):
        """Compute FP vectors from SMILES, apply stored scaler stats, then encode."""
        from data.fp_features import compute_raw_fp, build_fp_tensor

        desc_mean = self.desc_mean.cpu().numpy()
        desc_std = self.desc_std.cpu().numpy()

        tensors = []
        for smiles in smiles_list:
            result = compute_raw_fp(smiles)
            if result is None:
                tensors.append(torch.zeros(self.fp_dim))
            else:
                fp_bits, desc_vals = result
                tensors.append(build_fp_tensor(fp_bits, desc_vals, desc_mean, desc_std))

        fp_vec = torch.stack(tensors).to(device)
        with torch.no_grad():
            z = self.encode(fp_vec)
        return z

    def decode_to_smiles(self, z, charset):
        with torch.no_grad():
            logits = self.decode(z)
        return [onehot_to_smiles(logits[i], charset) for i in range(logits.shape[0])]
