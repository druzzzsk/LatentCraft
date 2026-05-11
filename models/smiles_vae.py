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
    """
    Autoregressive GRU decoder with teacher forcing.

    During training (x provided): input[t] = ground-truth character[t-1].
    During inference (x=None):    input[t] = argmax of predicted logit[t-1].
    Hidden state is initialised from z via a linear projection.
    """
    def __init__(self, max_len, n_chars, gru_depth, gru_dim, hidden_dim):
        super().__init__()
        self.max_len = max_len
        self.n_chars = n_chars
        self.gru_dim = gru_dim
        self.gru_depth = gru_depth

        # Project latent z to the initial hidden state of all GRU layers
        self.fc_z_to_h = nn.Linear(hidden_dim, gru_dim * gru_depth)

        # GRU receives one-hot of the previous character at every step
        self.gru = nn.GRU(
            input_size=n_chars,
            hidden_size=gru_dim,
            num_layers=gru_depth,
            batch_first=True,
        )
        self.fc_out = nn.Linear(gru_dim, n_chars)

    def _init_hidden(self, z):
        batch = z.shape[0]
        h = torch.tanh(self.fc_z_to_h(z))               # (batch, gru_dim * gru_depth)
        h = h.view(batch, self.gru_depth, self.gru_dim)  # (batch, depth, dim)
        return h.permute(1, 0, 2).contiguous()            # (depth, batch, dim)

    def forward(self, z, x=None):
        """
        z : (batch, hidden_dim)
        x : (batch, max_len, n_chars)  — ground truth for teacher forcing.
            Pass None during inference to use autoregressive decoding.
        Returns logits: (batch, max_len, n_chars)
        """
        batch = z.shape[0]
        h = self._init_hidden(z)

        if x is not None:
            # Teacher forcing: shift ground truth right by one step.
            # input[t] = x[t-1]; prepend a zero start-token.
            start = torch.zeros(batch, 1, self.n_chars, device=z.device)
            x_in = torch.cat([start, x[:, :-1, :]], dim=1)  # (batch, max_len, n_chars)
            out, _ = self.gru(x_in, h)                       # (batch, max_len, gru_dim)
            return self.fc_out(out)                           # (batch, max_len, n_chars)
        else:
            # Autoregressive: generate one character at a time
            outputs = []
            inp = torch.zeros(batch, 1, self.n_chars, device=z.device)
            for _ in range(self.max_len):
                out_t, h = self.gru(inp, h)          # (batch, 1, gru_dim)
                logit_t = self.fc_out(out_t)          # (batch, 1, n_chars)
                outputs.append(logit_t)
                # Greedy next input (no grad needed here)
                inp = torch.zeros(batch, 1, self.n_chars, device=z.device)
                inp.scatter_(2, logit_t.argmax(dim=2, keepdim=True), 1.0)
            return torch.cat(outputs, dim=1)          # (batch, max_len, n_chars)


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
        # Pass x for teacher forcing during training
        x_recon = self.decoder(z, x)
        return x_recon, z_mean, z_log_var

    def encode(self, x):
        z_mean, _ = self.encoder(x)
        return z_mean

    def decode(self, z):
        # Autoregressive inference — no ground truth provided
        return self.decoder(z)

    def sample(self, n, device="cpu"):
        return torch.randn(n, self.hidden_dim, device=device)

    def loss_function(self, x, x_recon_logits, z_mean, z_log_var, kl_weight=1.0):
        # x: (batch, max_len, n_chars) one-hot targets
        # x_recon_logits: (batch, max_len, n_chars) raw logits
        batch = x.shape[0]

        targets = x.argmax(dim=-1).reshape(-1)        # (batch * max_len,)
        logits = x_recon_logits.reshape(-1, self.n_chars)
        recon_loss = F.cross_entropy(logits, targets, reduction="sum") / batch

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        )

        total = recon_loss + kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def encode_smiles(self, smiles_list, charset, device="cpu"):
        tensors = [smiles_to_onehot(s, self.max_len, charset) for s in smiles_list]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            z = self.encode(x)
        return z

    def decode_to_smiles(self, z, charset):
        with torch.no_grad():
            logits = self.decode(z)
        smiles_list = []
        for i in range(logits.shape[0]):
            smiles_list.append(onehot_to_smiles(logits[i], charset))
        return smiles_list
