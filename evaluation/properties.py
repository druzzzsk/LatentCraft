from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")


def _to_mol(smiles):
    return Chem.MolFromSmiles(smiles) if smiles else None


def compute_logp(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)


def compute_qed(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return QED.qed(mol)


def compute_sa(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def compute_penalized_logp(smiles):
    """Penalized logP = logP - SA - cycle_penalty.

    This is the standard benchmark objective from Jin et al. (ICML 2018):
    cycle_penalty counts rings with more than 6 atoms.
    """
    mol = _to_mol(smiles)
    if mol is None:
        return None
    logp = Crippen.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    ring_info = mol.GetRingInfo()
    cycle_penalty = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
    return logp - sa - cycle_penalty


def compute_all(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return {
        "logP": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "SA": sascorer.calculateScore(mol),
        "penalized_logP": compute_penalized_logp(smiles),
    }
