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


def compute_all(smiles):
    mol = _to_mol(smiles)
    if mol is None:
        return None
    return {
        "logP": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "SA": sascorer.calculateScore(mol),
    }
