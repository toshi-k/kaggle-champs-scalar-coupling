from pathlib import Path
import numpy as np
import pandas as pd


def main():

    targets = list()
    targets.extend(list(Path('../02_triplet_update/_submission').iterdir()))
    targets.extend(list(Path('../03_edge_update/_submission').iterdir()))
    print('targets')
    print(targets)

    submissions = [pd.read_csv(target).sort_values('id') for target in targets]

    sub_concat = np.concatenate([sub[['scalar_coupling_constant']].values for sub in submissions], axis=1)

    new_sub = sub_concat.mean(axis=1)

    dir_sub = Path('_submission')
    dir_sub.mkdir(exist_ok=True)

    sub_df = pd.DataFrame()
    sub_df['id'] = submissions[0]['id']
    sub_df['scalar_coupling_constant'] = new_sub

    sub_df.to_csv(dir_sub / 'ensemble.csv', index=False)


if __name__ == '__main__':
    main()
