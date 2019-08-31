import random
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(seed):

    dir_dataset = Path('../../dataset/')

    train = pd.merge(pd.read_csv(dir_dataset / 'train.csv'),
                     pd.read_csv(dir_dataset / 'scalar_coupling_contributions.csv'))
    test = pd.read_csv(dir_dataset / 'test.csv')

    counts = train['molecule_name'].value_counts().sort_index()
    moles = list(counts.index)

    np.random.seed(seed)
    random.seed(seed)

    random.shuffle(moles)

    num_train = int(len(moles) * 0.9)
    train_moles = moles[:num_train]
    valid_moles = moles[num_train:]

    valid = train.query('molecule_name not in @train_moles')
    train = train.query('molecule_name in @train_moles')

    return train, valid, test, train_moles, valid_moles
