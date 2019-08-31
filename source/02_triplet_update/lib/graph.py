import numpy as np
import pandas as pd
from scipy.spatial import distance


class Graph:

    def __init__(self, points_df, bonds, list_atoms, charges):

        self.points = points_df[['x', 'y', 'z']].values
        self.bonds = bonds

        self.charges = charges[['eem', 'mmff94', 'gasteiger', 'qeq', 'qtpie',
                                'eem2015ha', 'eem2015hm', 'eem2015hn',
                                'eem2015ba', 'eem2015bm', 'eem2015bn']].values

        self.dists = distance.cdist(self.points, self.points)
        self.add_features = np.asarray(points_df[['in_ring', 'aromatic']].values, dtype=np.float32)

        self.adj = self.dists < 1.5
        self.num_nodes = len(points_df)
        self.connect_table = list(np.arange(self.num_nodes)[r] for r in self.adj)

        self.atoms = points_df['atom']
        self.dict_atoms = {at: i for i, at in enumerate(list_atoms)}

        self.angles, self.areas = self.calc_angles_areas()

    def get_atoms_array(self):

        atom_index = [self.dict_atoms[atom] for atom in self.atoms]
        one_hot = np.identity(len(self.dict_atoms))[atom_index]

        bond = np.sum(self.adj, 1) - 1
        bonds = np.identity(len(self.dict_atoms))[bond - 1]

        return np.concatenate([one_hot, bonds, self.add_features, self.charges], axis=1).astype(np.float32)

    def get_bond_features(self):

        bond_types = np.zeros((self.num_nodes, self.num_nodes, 5))
        bond_aromatic = np.zeros((self.num_nodes, self.num_nodes, 1))
        bond_in_ring = np.zeros((self.num_nodes, self.num_nodes, 1))
        bond_conjugated = np.zeros((self.num_nodes, self.num_nodes, 1))

        for i, row in self.bonds.iterrows():
            bond_types[row['atom_index_0'], row['atom_index_1'], row['bond_type']] = 1
            bond_types[row['atom_index_1'], row['atom_index_0'], row['bond_type']] = 1
            if row['aromatic']:
                bond_aromatic[row['atom_index_1'], row['atom_index_0'], 0] = 1
                bond_aromatic[row['atom_index_0'], row['atom_index_1'], 0] = 1
            if row['in_ring']:
                bond_in_ring[row['atom_index_1'], row['atom_index_0'], 0] = 1
                bond_in_ring[row['atom_index_0'], row['atom_index_1'], 0] = 1
            if row['conjugated']:
                bond_conjugated[row['atom_index_1'], row['atom_index_0'], 0] = 1
                bond_conjugated[row['atom_index_0'], row['atom_index_1'], 0] = 1

        return np.concatenate([bond_types, bond_aromatic, bond_in_ring, bond_conjugated], axis=2).astype(np.float32)

    def get_dists(self):
        return self.dists.astype(np.float32)

    def calc_angles_areas(self):

        list_angles = list()
        list_areas = list()

        for i in range(len(self.points)):
            diff = self.points - self.points[i]
            ip = np.dot(diff, diff.T)
            norms = np.apply_along_axis(np.linalg.norm, 1, diff)
            prod_norms = np.expand_dims(norms, 1) @ np.expand_dims(norms, 0)
            cos_theta = ip / prod_norms

            theta = np.arccos(cos_theta)
            theta[np.isnan(theta)] = 3.5
            list_angles.append(theta)

            sin_theta = np.sqrt(- cos_theta ** 2 + 1)
            sin_theta[np.isnan(sin_theta)] = 0.0
            area = prod_norms * sin_theta * 0.5
            list_areas.append(area)

        angles = np.stack(list_angles, axis=0).astype(np.float32)
        areas = np.stack(list_areas, axis=0).astype(np.float32)

        return angles, areas

    def get_angles(self):
        return self.angles

    def get_areas(self):
        return self.areas

    def show(self):
        print('\npoint')
        print(self.points)
        print('\nconnect table')
        print(self.connect_table)
        print('\natoms')
        print(self.atoms)


def main():

    structures = pd.read_csv('../../../input/structures.csv')
    strs_gp = structures.groupby('molecule_name')

    bonds = pd.read_csv('../../../input/bonds.csv')
    bonds_gp = bonds.groupby('molecule_name')

    train_charges = pd.read_csv('../../../input/train_ob_charges.csv')
    train_charges_gp = train_charges.groupby('molecule_name')

    list_atoms = list(set(structures['atom']))

    g = Graph(strs_gp.get_group('dsgdb9nsd_000001'),
              bonds_gp.get_group('dsgdb9nsd_000001'),
              list_atoms,
              train_charges_gp.get_group('dsgdb9nsd_000001'))
    g.show()

    print(g.get_atoms_array())
    print(g.get_angles())
    print(g.get_areas())
    a = g.get_areas()
    print(np.max(a))
    print(np.min(a))


if __name__ == '__main__':
    main()
