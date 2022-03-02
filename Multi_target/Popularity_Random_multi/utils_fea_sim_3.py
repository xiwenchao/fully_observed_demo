# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from config_3 import global_config as cfg
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys

from FM_old import FactorizationMachine
v = 0.2
d = 64

def feature_similarity(given_preference, userID, TopKTaxo):
    # TODO: We are using user embedding
    preference_matrix_all = cfg.user_emb[[userID]]
    # print('user_emb shape = {}'.format(preference_matrix_all.shape))
    # print('given_preference = {}'.format(given_preference))
    if len(given_preference) > 0:
        if len(given_preference) == 1:
            preference_matrix = cfg.emb_matrix[given_preference].reshape(1, d)
        else:
            preference_matrix = cfg.emb_matrix[given_preference]

        preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)

    total_result = list()
    result_dict = dict()

    for index, big_feature in enumerate(cfg.FACET_POOL):

        '''
        sampled_TS_matrix = [np.random.multivariate_normal(mean=big_feature_matrix[i].reshape(d), cov=v * v * cfg.facet_TS_matrix_inv[i + left])
                             for i in range(len(big_feature_matrix))]
        '''
        big_feature_matrix = cfg.emb_matrix[[index]]
        cosine_result = cosine_similarity(big_feature_matrix, preference_matrix_all)
        # cosine_result = cosine_result.sum(dim=0).reshape(cosine_result.shape[0], -1)
        cosine_result = np.sum(cosine_result, axis=1)
        cosine_result = -np.sort(-cosine_result)  # Sort it descending
        total_result.append((cosine_result))
        result_dict[big_feature] = np.sum(cosine_result[:TopKTaxo]) / (len(given_preference) + 1)  # choose top 5, normalization
        # TODO uncomment this for real value
        # result_dict[big_feature] = 2

    return result_dict


def feature_similarity_micro(given_preference, residual, userID):
    # TODO: We are using user embedding
    preference_matrix_all = cfg.user_emb[[userID]]
    if len(given_preference) > 0:
        preference_matrix = cfg.emb_matrix[given_preference]
        preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)

    base = cfg.star_count + cfg.price_count + cfg.city_count
    right = cfg.feature_count

    given_preference_ = [item - cfg.category_count for item in given_preference if item > base]

    to_test_feature = list(set(range(cfg.category_count)) - set(given_preference_))

    the_dict = dict()
    for item in to_test_feature:
        to_test_matrix = cfg.emb_matrix[[item + base]]
        cosine_result = cosine_similarity(to_test_matrix, preference_matrix_all)
        cosine_result = np.sum(cosine_result, axis=1)
        the_dict[item] = cosine_result

    value = sorted([v for k, v in the_dict.items()], reverse=True)
    position = [value.index(the_dict[big_f]) for big_f in residual]

    return position
