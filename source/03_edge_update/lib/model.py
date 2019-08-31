from functools import partial
import numpy as np
import pandas as pd

import chainer
from chainer import functions
from chainer import functions as F
from chainer.links import Linear
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

    def __init__(self, C):
        super(EdgeUpdate, self).__init__()
        with self.init_scope():
            self.W1 = ElementLinear(2 * C, nobias=True)
            self.W2 = ElementLinear(C, nobias=True)
            self.bn = ElementLayerNormalization(C)

    def __call__(self, edge, h):
        num_atom = edge.shape[1]
        h1 = F.tile(F.expand_dims(h, 1), (1, num_atom, 1, 1))
        h2 = F.tile(F.expand_dims(h, 2), (1, 1, num_atom, 1))
        concat = F.concat([h1, h2, edge], axis=3)

        add = zero_plus(self.W2(zero_plus(self.W1(concat))))

        return edge + self.bn(add)


class InteractionNetwork(chainer.Chain):

    def __init__(self, C):
        super(InteractionNetwork, self).__init__()
        with self.init_scope():
            self.W1 = ElementLinear(C, nobias=True)
            self.W2 = ElementLinear(C, nobias=True)
            self.W3 = ElementLinear(C, nobias=True)
            self.W4 = ElementLinear(C, nobias=True)
            self.W5 = ElementLinear(C, nobias=True)
            self.bn = ElementLayerNormalization(C)

    def __call__(self, h, edge):
        mt = zero_plus(self.W3(zero_plus(self.W2(edge))))
        mt = self.W1(h) * F.sum(mt, axis=1)
        h_add = self.W5(zero_plus(self.W4(mt)))
        return h + self.bn(h_add)


class EdgeUpdateNet(chainer.Chain):

    def __init__(self, num_layer, node_dim, edge_dim, gpu=0):
        super(EdgeUpdateNet, self).__init__()

        self.num_layer = num_layer
        self.edge_dim = edge_dim
        self.to_xpu = partial(to_device, gpu)

        with self.init_scope():
            self.gn = ElementLinear(node_dim)

            for layer in range(self.num_layer):
                self.add_link('eup{}'.format(layer), EdgeUpdate(edge_dim))
                self.add_link('int{}'.format(layer), InteractionNetwork(node_dim))

            self.interaction1 = Linear(512)
            self.interaction2 = Linear(512)
            self.interaction3 = Linear(4)

    def __call__(self, list_g, list_y):

        out = self.predict(list_g, list_y)

        yv = np.concatenate([y[['fc', 'sd', 'pso', 'dso']].values.astype(np.float32) for y in list_y], axis=0)
        yv_gpu = self.to_xpu(yv)

        return F.mean_absolute_error(out, yv_gpu) * 4 * len(list_y)

    def forward(self, list_g):

        input_array = F.stack([self.to_xpu(g.get_atoms_array().astype(np.float32)) for g in list_g], 0)
        dists = self.to_xpu(np.stack([g.get_dists() for g in list_g], 0).astype(np.float32))

        num_atom = dists.shape[1]

        num_rbf = self.edge_dim
        gamma = 20.0

        list_dists_rbf = list()

        embedlist = self.to_xpu(np.linspace(0, 10, num_rbf - 8, dtype=self.xp.float32))

        for g in list_g:

            dist = F.expand_dims(self.to_xpu(g.get_dists()), 0)
            dists_rbf = functions.reshape(dist, (1, num_atom, num_atom, 1))
            dists_rbf = functions.broadcast_to(dists_rbf, (1, num_atom, num_atom, num_rbf - 8))
            dists_rbf = functions.exp(- gamma * (dists_rbf - embedlist) ** 2)

            bond_feature = F.expand_dims(self.to_xpu(g.get_bond_features()), 0)

            list_dists_rbf.append(F.concat([dists_rbf, bond_feature], axis=3))

        e = F.concat(list_dists_rbf, axis=0)
        h = self.gn(input_array)

        for layer in range(self.num_layer):

            e = self['eup{}'.format(layer)](e, h)
            h = self['int{}'.format(layer)](h, e)

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

    train = pd.read_csv('../../../input/train2.csv')
    train_gp = train.groupby('molecule_name')

    train_charges = pd.read_csv('../../../input/train_ob_charges.csv')
    train_charges_gp = train_charges.groupby('molecule_name')

    list_atoms = list(set(structures['atom']))
    print(list_atoms)

    model = EdgeUpdateNet(num_layer=10, node_dim=512, edge_dim=512)
    model.to_gpu()

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
