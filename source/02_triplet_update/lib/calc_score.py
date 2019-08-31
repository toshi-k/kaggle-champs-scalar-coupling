from logging import getLogger

import numpy as np


def calc_score(df_truth, pred):

    logger = getLogger('root')

    target_types = list(set(df_truth['type']))

    diff = df_truth['scalar_coupling_constant'] - np.sum(pred, axis=1)
    diff_fc = df_truth['fc'] - pred[:, 0]
    diff_sd = df_truth['sd'] - pred[:, 1]
    diff_pso = df_truth['pso'] - pred[:, 2]
    diff_dso = df_truth['dso'] - pred[:, 3]

    scores = 0
    scores_exp = 0
    scores_fc = 0
    scores_sd = 0
    scores_pso = 0
    scores_dso = 0

    for target_type in target_types:

        target_pair = df_truth['type'] == target_type
        score_exp = np.mean(np.abs(diff[target_pair]))

        scores_exp += score_exp
        scores += np.log(score_exp)

        scores_fc += np.mean(np.abs(diff_fc[target_pair]))
        scores_sd += np.mean(np.abs(diff_sd[target_pair]))
        scores_pso += np.mean(np.abs(diff_pso[target_pair]))
        scores_dso += np.mean(np.abs(diff_dso[target_pair]))

    logger.info('valid mae (fc): {:.3f}'.format(scores_fc / len(target_types)))
    logger.info('valid mae (sd): {:.3f}'.format(scores_sd / len(target_types)))
    logger.info('valid mae (pso): {:.3f}'.format(scores_pso / len(target_types)))
    logger.info('valid mae (dso): {:.3f}'.format(scores_dso / len(target_types)))
    logger.info('valid mae (ALL): {:.3f}'.format(scores_exp / len(target_types)))

    return scores / len(target_types)
