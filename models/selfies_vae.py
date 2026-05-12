import torch

from models.smiles_vae import SmilesVAE
from data.preprocessing_selfies import (
    smiles_to_selfies_tokens,
    selfies_tokens_to_onehot,
    onehot_to_smiles_via_selfies,
)


class SelfiesVAE(SmilesVAE):
    """
    Same Conv1D+GRU architecture as SmilesVAE but operates on SELFIES token vocabulary.
    Only encode_smiles and decode_to_smiles differ — the model itself is identical.
    """

    def encode_smiles(self, smiles_list, charset, device="cpu"):
        tensors = []
        for smiles in smiles_list:
            tokens = smiles_to_selfies_tokens(smiles)
            if tokens is None:
                tokens = []
            tensors.append(selfies_tokens_to_onehot(tokens, self.max_len, charset))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            z = self.encode(x)
        return z

    def decode_to_smiles(self, z, charset):
        with torch.no_grad():
            logits = self.decode(z)
        return [onehot_to_smiles_via_selfies(logits[i], charset) for i in range(logits.shape[0])]
