import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return _morgan_gen.GetFingerprint(mol)


def property_improvement(seed_smiles, optimized_smiles, property_fn):
    """Средний прирост свойства: mean(property(opt) - property(seed))."""
    deltas = []
    for seed, opt in zip(seed_smiles, optimized_smiles):
        v_seed = property_fn(seed)
        v_opt = property_fn(opt)
        if v_seed is not None and v_opt is not None:
            deltas.append(v_opt - v_seed)
    if not deltas:
        return None
    return float(np.mean(deltas))


def success_rate(seed_smiles, optimized_smiles, property_fn, threshold=0.0):
    """Доля пар, где прирост свойства > threshold."""
    total = 0
    success = 0
    for seed, opt in zip(seed_smiles, optimized_smiles):
        v_seed = property_fn(seed)
        v_opt = property_fn(opt)
        if v_seed is None or v_opt is None:
            continue
        total += 1
        if (v_opt - v_seed) > threshold:
            success += 1
    if total == 0:
        return None
    return success / total


def similarity_to_seed(seed_smiles, optimized_smiles):
    """Tanimoto similarity (Morgan FP) между seed и оптимизированной молекулой."""
    similarities = []
    for seed, opt in zip(seed_smiles, optimized_smiles):
        fp_seed = _morgan_fp(seed)
        fp_opt = _morgan_fp(opt)
        if fp_seed is None or fp_opt is None:
            similarities.append(None)
        else:
            sim = DataStructs.TanimotoSimilarity(fp_seed, fp_opt)  # type: ignore[arg-type]
            similarities.append(float(sim))
    return similarities


def pareto_front(improvements, similarities):
    """Индексы точек на Pareto-фронте (max improvement, max similarity)."""
    points = list(zip(improvements, similarities))
    n = len(points)
    on_front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (points[j][0] >= points[i][0] and points[j][1] >= points[i][1] and
                    (points[j][0] > points[i][0] or points[j][1] > points[i][1])):
                dominated = True
                break
        if not dominated:
            on_front.append(i)
    return on_front
