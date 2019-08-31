from functools import partial
import numpy as np
import pandas as pd

import chainer
from chainer import functions
from chainer import functions as F
from chainer import links as L
from chainer.dataset import to_device

from lib.graph import Graph


def zero_plus(x):
    return F.softplus(x) - 0.6931472


class ElementLayerNormalization(chainer.links.LayerNormalization):

    def __call__(self, x):

        shape = x.shape
        h = F.reshape(x, (-1, shape[-1]))
        h = super(ElementLayerNormalization, self).__call__(h)
        h = F.reshape(h, shape)

        return h


class ElementLinear(chainer.links.Linear):

    def __call__(self, x):

        shape = x.shape
        h = F.reshape(x, (-1, shape[-1]))
        h = super(ElementLinear, self).__call__(h)
        shape_after = shape[:-1] + (self.out_size,)
        h = F.reshape(h, shape_after)

        return h


class EdgeUpdate(chainer.Chain):

    def __init__(self, edge_dim, triplet_dim):
        super(EdgeUpdate, self).__init__()
        with self.init_scope():
            self.Wn1 = ElementLinear(edge_dim, nobias=True)
            self.We1 = ElementLinear(edge_dim, nobias=True)
            self.We2 = ElementLinear(edge_dim, nobias=True)
            self.Wt1 = ElementLinear(triplet_dim, nobias=True)
            self.Wt2 = ElementLinear(edge_dim, nobias=True)
            self.bn = ElementLayerNormalization(edge_dim)

    def __call__(self, edge, node, triplet):
        num_atom = edge.shape[1]

        hn1 = F.tile(F.expand_dims(self.Wn1(node), 1), (1, num_atom, 1, 1))
        hn2 = F.tile(F.expand_dims(self.Wn1(node), 2), (1, 1, num_atom, 1))

        ht1 = self.Wt2(F.sum(zero_plus(self.Wt1(triplet)), axis=1))

        concat = F.concat([hn1, hn2, ht1, edge], axis=3)

        add = zero_plus(self.We2(zero_plus(self.We1(concat))))

        return edge + self.bn(add)


class TripletUpdate(chainer.Chain):

    def __init__(self, triplet_dim):
        super(TripletUpdate, self).__init__()
        with self.init_scope():
            self.Wn1 = ElementLinear(triplet_dim, nobias=True)
            self.We1 = ElementLinear(triplet_dim, nobias=True)
            self.Wt1 = ElementLinear(triplet_dim, nobias=True)
            self.Wt2 = ElementLinear(triplet_dim, nobias=True)
            self.bn = ElementLayerNormalization(triplet_dim)

    def __call__(self, triplet, edge, node):
        num_atom = triplet.shape[1]

        he1 = F.tile(F.expand_dims(self.We1(edge), 1), (1, num_atom, 1, 1, 1))
        he2 = F.tile(F.expand_dims(self.We1(edge), 2), (1, 1, num_atom, 1, 1))
        he3 = F.tile(F.expand_dims(self.We1(edge), 3), (1, 1, 1, num_atom, 1))

        hn1 = F.tile(F.expand_dims(F.expand_dims(self.Wn1(node), 1), 1), (1, num_atom, num_atom, 1, 1))
        hn2 = F.tile(F.expand_dims(F.expand_dims(self.Wn1(node), 2), 1), (1, num_atom, 1, num_atom, 1))
        hn3 = F.tile(F.expand_dims(F.expand_dims(self.Wn1(node), 2), 2), (1, 1, num_atom, num_atom, 1))

        concat = F.concat([he1, he2, he3, hn1, hn2, hn3, triplet], axis=4)

        add = zero_plus(self.Wt2(zero_plus(self.Wt1(concat))))

        return triplet + self.bn(add)


class NodeUpdate(chainer.Chain):

    def __init__(self, node_dim, edge_dim, triplet_dim):
        super(NodeUpdate, self).__init__()
        with self.init_scope():
            self.Wn1 = ElementLinear(node_dim, nobias=True)
            self.Wn2 = ElementLinear(node_dim, nobias=True)
            self.We1 = ElementLinear(edge_dim, nobias=True)
            self.We2 = ElementLinear(node_dim, nobias=True)
            self.Wt1 = ElementLinear(triplet_dim, nobias=True)
            self.Wt2 = ElementLinear(node_dim, nobias=True)
            self.bn = ElementLayerNormalization(node_dim)

    def __call__(self, node, triplet, edge):

        eh = F.sum(zero_plus(self.We2(zero_plus(self.We1(edge)))), axis=1)
        ht1 = self.Wt2(F.sum(F.sum(zero_plus(self.Wt1(triplet)), axis=3), axis=2))

        concat = F.concat([eh, ht1, node], axis=2)

        add = zero_plus(self.Wn2(zero_plus(self.Wn1(concat))))

        return node + self.bn(add)


class TripletUpdateNet(chainer.Chain):

    def __init__(self, num_layer, node_dim, edge_dim, triplet_dim, gpu=0):
        super(TripletUpdateNet, self).__init__()

        self.num_layer = num_layer
        self.edge_dim = edge_dim
        self.triplet_dim = triplet_dim
        self.to_xpu = partial(to_device, gpu)

        with self.init_scope():
            self.gn = ElementLinear(node_dim)

            for layer in range(self.num_layer):
                self.add_link('int{}'.format(layer), NodeUpdate(node_dim, edge_dim, triplet_dim))
                self.add_link('eup{}'.format(layer), EdgeUpdate(edge_dim, triplet_dim))
                self.add_link('tup{}'.format(layer), TripletUpdate(triplet_dim))

            self.interaction1 = L.Linear(512)
            self.interaction2 = L.Linear(512)
            self.interaction3 = L.Linear(4)

    def __call__(self, list_g, list_y):

        out = self.predict(list_g, list_y)

        yv = np.concatenate([y[['fc', 'sd', 'pso', 'dso']].values.astype(np.float32) for y in list_y], axis=0)
        yv_gpu = self.to_xpu(yv)

        return F.mean_absolute_error(out, yv_gpu) * 4 * len(list_y)

    def get_e_init(self, list_g):
        num_rbf = self.edge_dim
        gamma = 20.0

        list_dists_rbf = list()

        embedlist = self.to_xpu(np.linspace(0, 5, num_rbf - 8, dtype=self.xp.float32))

        for g in list_g:

            dist = F.expand_dims(self.to_xpu(g.get_dists()), 0)
            num_atom = dist.shape[1]
            dists_rbf = functions.reshape(dist, (1, num_atom, num_atom, 1))
            dists_rbf = functions.broadcast_to(dists_rbf, (1, num_atom, num_atom, num_rbf - 8))
            dists_rbf = functions.exp(- gamma * (dists_rbf - embedlist) ** 2)

            bond_feature = F.expand_dims(self.to_xpu(g.get_bond_features()), 0)

            list_dists_rbf.append(F.concat([dists_rbf, bond_feature], axis=3))

        e = F.concat(list_dists_rbf, axis=0)
        return e

    def get_t_init(self, list_g):
        num_rbf = self.triplet_dim // 2
        gamma = 20.0

        list_dists_rbf = list()

        embedlist = self.xp.linspace(0, 3.5, num_rbf, dtype=self.xp.float32)

        for g in list_g:

            angle = F.expand_dims(self.to_xpu(g.get_angles()), 0)
            num_atom = angle.shape[1]
            dists_rbf = functions.reshape(angle, (1, num_atom, num_atom, num_atom, 1))
            dists_rbf = functions.broadcast_to(dists_rbf, (1, num_atom, num_atom, num_atom, num_rbf))
            dists_rbf = functions.exp(- gamma * (dists_rbf - embedlist) ** 2)
            list_dists_rbf.append(dists_rbf)

        t1 = F.concat(list_dists_rbf, axis=0)

        num_rbf = -(- self.triplet_dim // 2)
        gamma = 10.0

        list_dists_rbf = list()

        embedlist = self.xp.linspace(0, 5.0, num_rbf, dtype=self.xp.float32)

        for g in list_g:

            angle = F.expand_dims(self.to_xpu(g.get_areas()), 0)
            num_atom = angle.shape[1]
            dists_rbf = functions.reshape(angle, (1, num_atom, num_atom, num_atom, 1))
            dists_rbf = functions.broadcast_to(dists_rbf, (1, num_atom, num_atom, num_atom, num_rbf))
            dists_rbf = functions.exp(- gamma * (dists_rbf - embedlist) ** 2)
            list_dists_rbf.append(dists_rbf)

        t2 = F.concat(list_dists_rbf, axis=0)

        return F.concat([t1, t2], axis=4)

    def forward(self, list_g):

        input_array = F.stack([self.to_xpu(g.get_atoms_array().astype(np.float32)) for g in list_g], 0)

        e = self.get_e_init(list_g)
        h = self.gn(input_array)
        t = self.get_t_init(list_g)

        for layer in range(self.num_layer):

            t = self['tup{}'.format(layer)](t, e, h)
            h = self['int{}'.format(layer)](h, t, e)
            e = self['eup{}'.format(layer)](e, h, t)

        h_out = F.concat((h, input_array), axis=2)

        return h_out, e

    def predict(self, list_g, list_y):

        out, ko = self.forward(list_g)

        list_concat1 = list()
        list_concat2 = list()

        for i, (g, y) in enumerate(zip(list_g, list_y)):
            d = self.to_xpu(g.get_dists())
            dists = F.expand_dims(d[y['atom_index_0'].values, y['atom_index_1'].values], 1)

            s = F.concat((out[i, y['atom_index_0'].values, :],
                          out[i, y['atom_index_1'].values, :],
                          ko[i, y['atom_index_0'].values, y['atom_index_1'].values, :],
                          ko[i, y['atom_index_1'].values, y['atom_index_0'].values, :],
                          dists), axis=1)
            list_concat1.append(s)

            s = F.concat((out[i, y['atom_index_1'].values, :],
                          out[i, y['atom_index_0'].values, :],
                          ko[i, y['atom_index_1'].values, y['atom_index_0'].values, :],
                          ko[i, y['atom_index_0'].values, y['atom_index_1'].values, :],
                          dists), axis=1)
            list_concat2.append(s)

        concat1 = F.concat(list_concat1, axis=0)
        concat2 = F.concat(list_concat2, axis=0)

        h11 = F.leaky_relu(self.interaction1(concat1))
        h12 = F.leaky_relu(self.interaction2(h11))
        out1 = self.interaction3(h12)

        h21 = F.leaky_relu(self.interaction1(concat2))
        h22 = F.leaky_relu(self.interaction2(h21))
        out2 = self.interaction3(h22)

        return (out1 + out2) / 2.0


def main():

    structures = pd.read_csv('../../../input/structures.csv')
    strs_gp = structures.groupby('molecule_name')

    bonds = pd.read_csv('../../../input/bonds.csv')
    bonds_gp = bonds.groupby('molecule_name')

    train = pd.merge(pd.read_csv('../../../dataset/train.csv'),
                     pd.read_csv('../../../dataset/scalar_coupling_contributions.csv'))
    train_gp = train.groupby('molecule_name')

    list_atoms = list(set(structures['atom']))
    print(list_atoms)

    model = TripletUpdateNet(num_layer=4, node_dim=512, edge_dim=256, triplet_dim=128)
    model.to_gpu()

    train_charges = pd.read_csv('../../../input/train_ob_charges.csv')
    train_charges_gp = train_charges.groupby('molecule_name')

    target1 = 'dsgdb9nsd_000008'

    g1 = Graph(strs_gp.get_group(target1),
               bonds_gp.get_group(target1),
               list_atoms,
               train_charges_gp.get_group(target1))
    y1 = train_gp.get_group(target1)
    out = model([g1], [y1])
    print(out)

    target2 = 'dsgdb9nsd_000010'

    g2 = Graph(strs_gp.get_group(target2),
               bonds_gp.get_group(target2),
               list_atoms,
               train_charges_gp.get_group(target2))
    y2 = train_gp.get_group(target2)
    out = model([g2], [y2])
    print(out)

    out = model([g1, g2], [y1, y2])
    print(out)


if __name__ == '__main__':
    main()
