import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _to_mol(smiles):
    return Chem.MolFromSmiles(smiles) if smiles else None


def _canonical(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _morgan_fp(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return _morgan_gen.GetFingerprint(mol)


def validity(smiles_list):
    if not smiles_list:
        return 0.0
    valid = sum(1 for s in smiles_list if _to_mol(s) is not None)
    return valid / len(smiles_list)


def uniqueness(smiles_list):
    valid = [_canonical(s) for s in smiles_list]
    valid = [s for s in valid if s is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def novelty(smiles_list, train_smiles):
    train_set = set()
    for s in train_smiles:
        c = _canonical(s)
        if c:
            train_set.add(c)

    valid = [_canonical(s) for s in smiles_list]
    valid = [s for s in valid if s is not None]
    if not valid:
        return 0.0

    novel = sum(1 for s in valid if s not in train_set)
    return novel / len(valid)


def diversity(smiles_list, max_pairs=5000):
    """Средний попарный Tanimoto distance (1 - similarity) по Morgan FP.

    Для больших наборов берётся случайная подвыборка пар.
    """
    fps = [_morgan_fp(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0

    # Все пары или случайная подвыборка
    indices = list(range(len(fps)))
    pairs = [(i, j) for i in indices for j in indices if i < j]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    distances = []
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])  
        distances.append(1.0 - sim)

    return float(np.mean(distances))


def reconstruction_accuracy(original_smiles, reconstructed_smiles):
    if not original_smiles:
        return 0.0
    correct = 0
    for orig, recon in zip(original_smiles, reconstructed_smiles):
        c_orig = _canonical(orig)
        c_recon = _canonical(recon)
        if c_orig is not None and c_orig == c_recon:
            correct += 1
    return correct / len(original_smiles)
