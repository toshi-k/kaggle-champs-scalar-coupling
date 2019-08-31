import random

import numpy as np
import pandas as pd


def generate_batches(structures_groups, moles, batch_size, use_remainder=False):

    batches = list()

    atom_counts = pd.DataFrame([(mol, len(structures_groups.get_group(mol))) for mol in moles],
                               columns=['molecule_name', 'num_atom'])

    num_atom_counts = atom_counts['num_atom'].value_counts()

    for count, num_mol in num_atom_counts.to_dict().items():
        if use_remainder:
            num_batch_for_this = -(-num_mol // batch_size)
        else:
            num_batch_for_this = num_mol // batch_size

        target_mols = atom_counts.query('num_atom==@count')['molecule_name'].to_list()
        random.shuffle(target_mols)

        devider = np.arange(0, len(target_mols), batch_size)
        devider = np.append(devider, 99999)

        if use_remainder:
            target_mols = np.append(target_mols, np.repeat(target_mols[-1], -len(target_mols) % batch_size))

        for b in range(num_batch_for_this):
            batches.append(target_mols[devider[b]:devider[b+1]])

    return batches
