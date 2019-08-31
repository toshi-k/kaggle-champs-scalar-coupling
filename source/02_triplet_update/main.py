import random
import time
import argparse
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
from tqdm import tqdm

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

from lib.log import init_logger
from lib.graph import Graph
from lib.model import TripletUpdateNet
from lib.calc_score import calc_score
from lib.load_dataset import load_dataset
from lib.generate_batches import generate_batches
from lib.predicts import predicts, make_submission


def main():

    random.seed(params.seed)
    np.random.seed(params.seed)

    tic = time.time()
    logger = init_logger(Path('_log') / 'log_b{:d}_l{:d}_nd{:d}_seed{:d}.log'.format(
        params.batch_size, params.num_layer, params.node_dim, params.seed))

    logger.info('parameters')
    logger.info(vars(params))

    train, valid, test, train_moles, valid_moles = load_dataset(params.seed)

    valid_moles = sorted(valid_moles)
    valid.sort_values('molecule_name', inplace=True)

    logger.info('train moles: {} ...'.format(train_moles[:5]))
    logger.info('valid moles: {} ...'.format(valid_moles[:5]))

    test_moles = sorted(list(set(test['molecule_name'])))
    test.sort_values('molecule_name', inplace=True)

    logger.info('train data: {}'.format(train.shape))
    logger.info('valid data: {}'.format(valid.shape))
    logger.info('test data: {}'.format(test.shape))

    structures = pd.read_csv('../../input/structures.csv')
    structures_groups = structures.groupby('molecule_name')

    bonds = pd.read_csv('../../input/bonds.csv')
    bonds_gp = bonds.groupby('molecule_name')

    train_charges = pd.read_csv('../../input/train_ob_charges.csv')
    train_charges_gp = train_charges.groupby('molecule_name')

    test_charges = pd.read_csv('../../input/test_ob_charges.csv')
    test_charges_gp = test_charges.groupby('molecule_name')

    train_targets = train.groupby('molecule_name')
    valid_targets = valid.groupby('molecule_name')
    test_targets = test.groupby('molecule_name')

    if params.debug:
        random.shuffle(train_moles)
        train_moles = train_moles[:5000]
        test_moles = test_moles[:1000]

    valid.sort_values('id', inplace=True)
    test.sort_values('id', inplace=True)

    list_atoms = list(set(structures['atom']))

    train_graphs = dict()
    for mole in tqdm(train_moles):
        train_graphs[mole] = Graph(structures_groups.get_group(mole),
                                   bonds_gp.get_group(mole),
                                   list_atoms, train_charges_gp.get_group(mole))

    valid_graphs = dict()
    for mole in tqdm(valid_moles):
        valid_graphs[mole] = Graph(structures_groups.get_group(mole),
                                   bonds_gp.get_group(mole),
                                   list_atoms, train_charges_gp.get_group(mole))

    test_graphs = dict()
    for mole in tqdm(test_moles):
        test_graphs[mole] = Graph(structures_groups.get_group(mole),
                                  bonds_gp.get_group(mole),
                                  list_atoms, test_charges_gp.get_group(mole))

    model = TripletUpdateNet(num_layer=params.num_layer,
                             node_dim=params.node_dim, edge_dim=params.edge_dim, triplet_dim=params.triplet_dim,
                             gpu=params.gpu)
    if params.gpu >= 0:
        logger.info('transfer model to GPU {}'.format(params.gpu))
        model.to_gpu(params.gpu)

    optimizer = optimizers.Adam(alpha=5e-4)
    optimizer.setup(model)
    model.cleargrads()

    epoch = 2 if params.debug else params.epoch

    for ep in range(epoch):

        logger.info('')
        logger.info('')
        logger.info('start epoch {}'.format(ep))
        logger.info('')

        # -------------------------
        logger.info('')
        logger.info('training')

        loss_value = 0
        random.shuffle(train_moles)
        train_batches_moles = generate_batches(structures_groups, train_moles, params.batch_size)
        random.shuffle(train_batches_moles)

        for batch_moles in tqdm(train_batches_moles):

            list_train_X = list()
            list_train_y = list()

            for target_mol in batch_moles:
                list_train_X.append(train_graphs[target_mol])
                list_train_y.append(train_targets.get_group(target_mol))

            with chainer.using_config('train', ep == 0):

                loss = model(list_train_X, list_train_y)

                model.cleargrads()
                loss.backward()
                optimizer.update()

            loss_value += cuda.to_cpu(loss.data)

        logger.info('train loss: {:.3f}'.format(float(loss_value) / len(train_moles)))

        # -------------------------
        logger.info('')
        logger.info('validation')

        valid_df = predicts(structures_groups, valid_moles, valid_graphs, valid_targets, model, params.batch_size)

        valid_pred = valid_df[['fc', 'sd', 'pso', 'dso']]

        valid_score = calc_score(valid, valid_pred.values)
        logger.info('valid score: {:.3f}'.format(valid_score))

        # -------------------------

        optimizer.alpha = optimizer.alpha * 0.95

        logger.info('change learning rate: {:.6f}'.format(optimizer.alpha))

        if (ep + 1) % 20 == 0:

            # -------------------------
            # save model

            dir_model = Path('_model')
            logger.info('save model')
            dir_model.mkdir(exist_ok=True)
            serializers.save_npz(dir_model / 'model_ep{}_seed{}.npz'.format(ep, params.seed), model)

            # -------------------------
            # make submission

            logger.info('')
            logger.info('test')

            test_df = predicts(structures_groups, test_moles, test_graphs, test_targets, model, params.batch_size)
            make_submission(test_df, ep, valid_score, params.seed, dir_sub=Path('_submission'))
            make_submission(valid_df, ep, valid_score, params.seed, dir_sub=Path('_valid'))

    toc = time.time() - tic
    logger.info('Elapsed time {:.1f} [min]'.format(toc / 60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, type=strtobool)

    parser.add_argument('--epoch', default=100, type=int)

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--num_layer', default=4, type=int)

    parser.add_argument('--node_dim', default=512, type=int)

    parser.add_argument('--edge_dim', default=256, type=int)

    parser.add_argument('--triplet_dim', default=96, type=int)

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--seed', default=1048, type=int)

    params = parser.parse_args()

    main()
