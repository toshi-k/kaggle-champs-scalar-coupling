import numpy as np
import pandas as pd

from xyz2mol_jo import xyz2mol
from tqdm import tqdm


def main():

    atom2index = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    structures = pd.read_csv('../../dataset/structures.csv')
    structures_gp = structures.groupby('molecule_name')

    list_result = list()
    list_bonds = list()

    charged_fragments = True
    quick = True

    dict_hbd = {'S': 0, 'SP': 1, 'SP2': 2, 'SP3': 3}

    i = 0

    for name, group in tqdm(structures_gp):

        xyz_coordinates = group[['x', 'y', 'z']].values.tolist()
        atomicNumList = [atom2index[a] for a in group['atom']]

        # try:
        mol = xyz2mol(atomicNumList, 0, xyz_coordinates, charged_fragments, quick)
        num_atoms = mol.GetNumAtoms()

        list_degree = [mol.GetAtomWithIdx(i).GetDegree() for i in range(num_atoms)]
        list_is_ring = [mol.GetAtomWithIdx(i).IsInRing() for i in range(num_atoms)]
        list_is_ring3 = [mol.GetAtomWithIdx(i).IsInRingSize(3) for i in range(num_atoms)]
        list_is_ring4 = [mol.GetAtomWithIdx(i).IsInRingSize(4) for i in range(num_atoms)]
        list_is_ring5 = [mol.GetAtomWithIdx(i).IsInRingSize(5) for i in range(num_atoms)]
        list_is_ring6 = [mol.GetAtomWithIdx(i).IsInRingSize(6) for i in range(num_atoms)]
        list_is_ring7 = [mol.GetAtomWithIdx(i).IsInRingSize(7) for i in range(num_atoms)]
        list_is_ring8 = [mol.GetAtomWithIdx(i).IsInRingSize(8) for i in range(num_atoms)]
        list_is_aromatic = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(num_atoms)]
        list_is_hybridization = [dict_hbd[str(mol.GetAtomWithIdx(i).GetHybridization())] for i in range(num_atoms)]

        bonds = mol.GetBonds()
        for bond in bonds:
            bond_type = int(bond.GetBondType())
            bond_type = 4 if bond_type == 12 else bond_type

            list_bonds.append(
                {'molecule_name': name,
                 'atom_index_0': int(bond.GetBeginAtomIdx()),
                 'atom_index_1': int(bond.GetEndAtomIdx()),
                 'aromatic': int(bond.GetIsAromatic()),
                 'in_ring': int(bond.IsInRing()),
                 'bond_type': bond_type,
                 'conjugated': int(bond.GetIsConjugated()),
                 }
            )

        new_result = group.copy()
        new_result['degree'] = np.array(list_degree, dtype=np.int32)
        new_result['in_ring'] = np.array(list_is_ring, dtype=np.int32)
        new_result['in_ring3'] = np.array(list_is_ring3, dtype=np.int32)
        new_result['in_ring4'] = np.array(list_is_ring4, dtype=np.int32)
        new_result['in_ring5'] = np.array(list_is_ring5, dtype=np.int32)
        new_result['in_ring6'] = np.array(list_is_ring6, dtype=np.int32)
        new_result['in_ring7'] = np.array(list_is_ring7, dtype=np.int32)
        new_result['in_ring8'] = np.array(list_is_ring8, dtype=np.int32)
        new_result['aromatic'] = np.array(list_is_aromatic, dtype=np.int32)
        new_result['hybridization'] = np.array(list_is_hybridization, dtype=np.int32)

        list_result.append(new_result)

        i += 1

        if i > 1000:
            break

    result_all = pd.concat(list_result, axis=0)
    result_all.to_csv('../../input/structures.csv', index=False)

    result_bonds = pd.DataFrame(list_bonds)
    result_bonds.to_csv('../../input/bonds.csv', index=False)

    print(result_bonds['bond_type'].value_counts())


if __name__ == "__main__":
    main()
