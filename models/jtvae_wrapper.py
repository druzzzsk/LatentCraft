import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'jtvae'))

from fast_jtnn import JTNNVAE, Vocab
from fast_jtnn.nnutils import set_device
from fast_jtnn.datautils import tensorize
from fast_jtnn.mol_tree import MolTree


class JTVAEWrapper:
    """
    Wrapper around JTNNVAE with the same interface as SmilesVAE:
      encode_smiles(smiles_list, charset=None, device='cpu') -> z  (batch, latent_size*2)
      decode_to_smiles(z, charset=None)                         -> list[str | None]
      sample(n, device='cpu')                                   -> z  (n, latent_size*2)

    The `charset` argument is accepted for API compatibility but is ignored —
    JT-VAE uses junction trees, not character sets.

    Latent vector layout: z = [z_tree | z_mol], each half of size `latent_size`.
    """

    def __init__(self, vocab_path, checkpoint_path,
                 hidden_size=450, latent_size=56, depthT=20, depthG=3):
        vocab_words = [x.strip() for x in open(vocab_path)]
        self.vocab = Vocab(vocab_words)
        self.model = JTNNVAE(
            self.vocab,
            hidden_size=hidden_size,
            latent_size=latent_size,
            depthT=depthT,
            depthG=depthG,
        )
        state = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()
        self.latent_size = self.model.latent_size  # per-half size (latent_size // 2)
        self.hidden_dim = latent_size  # full latent dim for downstream compatibility

    def to(self, device):
        self.model = self.model.to(device)
        set_device(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode_smiles(self, smiles_list, charset=None, device='cpu'):
        """Encode a list of SMILES into latent vectors (mean, no reparameterization noise)."""
        self.model = self.model.to(device)
        set_device(device)
        self.model.eval()

        # Filter out SMILES that fail to parse into MolTree
        valid_trees = []
        valid_idx = []
        for i, s in enumerate(smiles_list):
            try:
                tree = MolTree(s)
                if len(tree.nodes) > 0:
                    valid_trees.append(tree)
                    valid_idx.append(i)
            except Exception:
                pass

        if not valid_trees:
            # Return zeros for all if none parsed
            return torch.zeros(len(smiles_list), self.latent_size * 2, device=device)

        _, jtenc_holder, mpn_holder = tensorize(valid_trees, self.vocab, assm=False)
        with torch.no_grad():
            z_mean, _ = self.model.encode_latent(jtenc_holder, mpn_holder)

        # If some SMILES failed, fill their slots with zeros
        if len(valid_idx) < len(smiles_list):
            result = torch.zeros(len(smiles_list), z_mean.shape[1], device=device)
            for out_i, orig_i in enumerate(valid_idx):
                result[orig_i] = z_mean[out_i]
            return result

        return z_mean.to(device)

    def decode_to_smiles(self, z, charset=None):
        """Decode a batch of latent vectors to SMILES strings (None if decoding fails)."""
        self.model.eval()
        device = z.device
        results = []
        for i in range(z.shape[0]):
            z_i = z[i].unsqueeze(0)
            z_tree = z_i[:, :self.latent_size]
            z_mol = z_i[:, self.latent_size:]
            try:
                with torch.no_grad():
                    smiles = self.model.decode(z_tree, z_mol, prob_decode=False)
                results.append(smiles)
            except Exception:
                results.append(None)
        return results

    def sample(self, n, device='cpu'):
        """Sample n random latent vectors from the prior N(0, I)."""
        return torch.randn(n, self.latent_size * 2, device=device)


def load_jtvae(vocab_path, checkpoint_path,
               hidden_size=450, latent_size=56, depthT=20, depthG=3,
               device='cpu'):
    """Convenience loader — returns a ready-to-use JTVAEWrapper."""
    wrapper = JTVAEWrapper(
        vocab_path=vocab_path,
        checkpoint_path=checkpoint_path,
        hidden_size=hidden_size,
        latent_size=latent_size,
        depthT=depthT,
        depthG=depthG,
    )
    wrapper.to(device)
    return wrapper
