"""
This script is fork from Jo's public notebook
https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization
Thank you !
"""

# rdkit & xyz2mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions  # Only needed if modifying defaults
from rdkit.Chem.rdmolops import SanitizeFlags

# https://github.com/jensengroup/xyz2mol
from xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path

CACHEDIR = Path('./')


def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol, -1)

    return mol


def xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick):
    AC, mol = xyz2AC(atomicNumList, xyz_coordinates)
    new_mol = AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol
