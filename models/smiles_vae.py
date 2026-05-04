import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocessing import smiles_to_onehot, onehot_to_smiles


class Encoder(nn.Module):
    def __init__(self, max_len, n_chars, conv_depth, conv_start_filters, hidden_dim):
        super().__init__()

        layers = []
        in_ch = n_chars
        out_ch = conv_start_filters
        for _ in range(conv_depth):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4),
                nn.BatchNorm1d(out_ch),
                nn.Tanh(),
            ]
            in_ch = out_ch
            out_ch = out_ch * 2

        self.conv = nn.Sequential(*layers)

        conv_out_dim = in_ch * max_len
        mid_dim = conv_out_dim // 2

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, mid_dim),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(mid_dim, hidden_dim)
        self.fc_log_var = nn.Linear(mid_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, max_len, n_chars)
        x = x.permute(0, 2, 1)   # -> (batch, n_chars, max_len)
        x = self.conv(x)
        x = x.flatten(1)          # -> (batch, conv_out_dim)
        x = self.fc(x)
        return self.fc_mean(x), self.fc_log_var(x)


class Decoder(nn.Module):
    def __init__(self, max_len, n_chars, gru_depth, gru_dim, hidden_dim):
        super().__init__()
        self.max_len = max_len
        self.n_chars = n_chars
        self.gru_dim = gru_dim

        self.fc_in = nn.Linear(hidden_dim, gru_dim)
        self.gru = nn.GRU(
            input_size=gru_dim,
            hidden_size=gru_dim,
            num_layers=gru_depth,
            batch_first=True,
        )
        self.fc_out = nn.Linear(gru_dim, n_chars)

    def forward(self, z):
        # z: (batch, hidden_dim)
        h = torch.tanh(self.fc_in(z))        # (batch, gru_dim)
        # replicate across max_len timesteps
        x = h.unsqueeze(1).expand(-1, self.max_len, -1)  # (batch, max_len, gru_dim)
        out, _ = self.gru(x)                 # (batch, max_len, gru_dim)
        logits = self.fc_out(out)            # (batch, max_len, n_chars)
        return logits


class SmilesVAE(nn.Module):
    def __init__(self, max_len, n_chars, hidden_dim=196, conv_depth=4,
                 conv_start_filters=16, gru_depth=4, gru_dim=488):
        super().__init__()
        self.max_len = max_len
        self.n_chars = n_chars
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(max_len, n_chars, conv_depth, conv_start_filters, hidden_dim)
        self.decoder = Decoder(max_len, n_chars, gru_depth, gru_dim, hidden_dim)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def encode(self, x):
        # Returns z_mean (no noise) -- shape (batch, hidden_dim)
        z_mean, _ = self.encoder(x)
        return z_mean

    def decode(self, z):
        # Returns logits: (batch, max_len, n_chars)
        return self.decoder(z)

    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.hidden_dim, device=device)
        return z

    def loss_function(self, x, x_recon_logits, z_mean, z_log_var, kl_weight=1.0):
        # x: (batch, max_len, n_chars) -- one-hot targets
        # x_recon_logits: (batch, max_len, n_chars) -- raw logits
        batch = x.shape[0]

        # Reconstruction: cross-entropy over character positions
        targets = x.argmax(dim=-1).reshape(-1)    # (batch * max_len,)
        logits = x_recon_logits.reshape(-1, self.n_chars)
        recon_loss = F.cross_entropy(logits, targets, reduction="sum") / batch

        # KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        )

        total = recon_loss + kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def encode_smiles(self, smiles_list, charset, device="cpu"):
        """Convenience: list of SMILES strings -> z_mean tensor."""
        tensors = [smiles_to_onehot(s, self.max_len, charset) for s in smiles_list]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            z = self.encode(x)
        return z

    def decode_to_smiles(self, z, charset):
        """Convenience: z tensor -> list of SMILES strings."""
        with torch.no_grad():
            logits = self.decode(z)
        smiles_list = []
        for i in range(logits.shape[0]):
            smiles_list.append(onehot_to_smiles(logits[i], charset))
        return smiles_list
