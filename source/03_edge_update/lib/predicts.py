from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import chainer
from chainer import cuda

from lib.generate_batches import generate_batches


def make_submission(test_df, epoch, valid_score, seed, dir_sub=Path('_submission')):

    submission = pd.DataFrame(test_df[['id_temp', 'scalar_coupling_constant']])
    submission.columns = ['id', 'scalar_coupling_constant']

    dir_sub.mkdir(exist_ok=True)
    submission.to_csv(dir_sub / 'edge_update_ep{}_valid{:.3f}_seed{}.csv'.format(epoch, valid_score, seed),
                      index=False)


def predicts(structures_groups, moles, graphs, targets, model, batch_size):

    valid_batches_moles = generate_batches(structures_groups, moles, batch_size, use_remainder=True)

    valid_pred_list = list()
    valid_target_list = list()

    for batch_moles in tqdm(valid_batches_moles):

        list_valid_X = list()
        list_valid_y = list()

        for target_mol in batch_moles:
            list_valid_X.append(graphs[target_mol])
            list_valid_y.append(targets.get_group(target_mol))

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            pred = model.predict(list_valid_X, list_valid_y)

        valid_target_list.extend(list_valid_y)
        valid_pred_list.append(cuda.to_cpu(pred.data))

    valid_df = pd.DataFrame(np.concatenate(valid_pred_list, axis=0), columns=['fc', 'sd', 'pso', 'dso'])
    valid_df['id_temp'] = pd.concat(valid_target_list, axis=0).reset_index(drop=True)['id']

    valid_df.drop_duplicates(subset='id_temp', inplace=True)
    valid_df.sort_values('id_temp', inplace=True)

    valid_df['scalar_coupling_constant'] = np.sum(valid_df[['fc', 'sd', 'pso', 'dso']].values, axis=1)
    return valid_df
