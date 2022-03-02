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

from collections import defaultdict

import argparse
import sys
from heapq import nlargest, nsmallest

from FM_old import FactorizationMachine
from config_3 import global_config as cfg
import operator

from utils_item_sim_3 import item_score


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def evaluate_change(static_score, another_score, TopK):
    assert len(static_score) == len(another_score)

    static_dict = dict()
    another_dict = dict()

    for index in range(len(static_score)):
        static_dict[static_score[index]] = index
        another_dict[another_score[index]] = index

    s = nlargest(TopK, static_score)
    a = nlargest(TopK, another_score)

    s_index = [static_dict[item] for item in s]
    a_index = [another_dict[item] for item in a]

    intersection = set(s_index).intersection(set(a_index))

    return len(intersection)


def rank_items(given_preference, userID, itemID, skip_big_feature, FM_model, candidate_list, write_fp, after_update, rej_list, previous_dict):
    FM_model.eval()
    start = time.time()
    #TODO: change later
    #candidate_list += rej_list
    #train_sample = cfg._train_user_to_items[str(userID)]
    #candidate_list += train_sample
    mini_ui_pair = np.zeros((len(candidate_list), 2))
    for index, itemID_ in enumerate(candidate_list):
        mini_ui_pair[index, :] = [userID, itemID_ + len(cfg.user_list)]  # TODO: Potentially a huge bug here!
    mini_ui_pair = torch.from_numpy(mini_ui_pair).long()
    mini_ui_pair = cuda_(mini_ui_pair)
    '''
    static_preference_index = torch.LongTensor(given_preference)
    static_preference_index = cuda_(static_preference_index)
    static_score = np.array(item_score(mini_ui_pair, static_preference_index))
    '''
    # static_score = static_score.detach().cpu().numpy()
    static_preference_index = torch.LongTensor(given_preference).expand(len(candidate_list), len(
        given_preference))  # candidate_list, given preference
    static_preference_index = cuda_(static_preference_index)
    static_score, _, _ = FM_model(mini_ui_pair, None, static_preference_index)
    static_score = static_score.detach().cpu().numpy()

    score_dict = dict()
    inverted_score_dict = dict()
    for index, item in enumerate(static_score.reshape(-1).tolist()):
        score_dict[item] = candidate_list[index]
        inverted_score_dict[candidate_list[index]] = item

    assert len(candidate_list) == len(static_score)
    assert candidate_list[-1] == itemID
    target_value = static_score[-1]
    #print('target score is: {}'.format(target_value))

    ranked_score = nlargest(100000, static_score.reshape(-1).tolist())
    max_item_score = ranked_score[0]
    #print('ranked 10 is: {}'.format(ranked_score[: 10]))
    target_position = ranked_score.index(target_value)

    if len(rej_list) > 0:
        rejected_score = static_score[-len(rej_list) - 1: -1]
        rej_position = []
        for score in rejected_score:
            rej_position.append(ranked_score.index(score))


    ranked_item = [score_dict[score] for score in ranked_score]  # It is the ranked items to return
    ranked_item = [item for item in ranked_item if item not in rej_list]

    predictions = static_score.reshape((len(static_score), 1)[0])
    y_true = [0] * len(predictions)
    y_true[-1] = 1
    tmp = list(zip(y_true, predictions))
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    y_true, predictions = zip(*tmp)
    if len(y_true) > 1:
        auc = roc_auc_score(y_true, predictions)
        #print('auc is: {}'.format(round(auc, 3)))
    #print('FM_inference takes: {} seconds'.format(time.time() - start))

    FM_model.train()

    #print('top 10: {}'.format(ranked_item[: 10]))
    #print('top 20: {}'.format(ranked_item[10: 20]))
    #print('top 10 score: {}'.format(ranked_score[: 10]))
    #print('top 20 score: {}'.format(ranked_score[10: 20]))

    return ranked_item, inverted_score_dict


def try_feature_cause_change(given_preference, userID, itemID, skip_big_feature, FM_model, candidate_list, write_fp):
    # TODO: We are using user embedding
    preference_matrix_all = cfg.user_emb[[userID, itemID]]  # Note that user embedding actually means ui_embs
    if len(given_preference) > 0:
        preference_matrix = cfg.emb_matrix[given_preference]
        preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)

    # TODO: One boosting point for calculation is: skip those big feature whose (1) entropy is zero, (2) has been asked
    start = time.time()
    big_small_mapping = dict()
    small_feature_raw_result_from_FM = dict()

    for index, big_feature in enumerate(cfg.FACET_POOL[: 3]):
        if big_feature in skip_big_feature:
            continue
        left = cfg.spans[index][0]
        right = cfg.spans[index][1]
        big_small_mapping[big_feature] = range(left, right)

    for index, big_feature in enumerate(cfg.FACET_POOL[3: ]):
        if big_feature in skip_big_feature:
            continue
        base = cfg.star_count + cfg.city_count + cfg.price_count
        feature_index = [item + base for item in cfg.taxo_dict[big_feature]]
        big_small_mapping[big_feature] = feature_index

    residual_feature = list()
    for k, v in big_small_mapping.items():
        residual_feature += v

    mini_ui_pair = np.zeros((len(candidate_list), 2))
    for index, itemID in enumerate(candidate_list):
        mini_ui_pair[index, :] = [userID, itemID]
    mini_ui_pair = torch.from_numpy(mini_ui_pair).long()
    mini_ui_pair = cuda_(mini_ui_pair)

    static_preference_index = torch.LongTensor(given_preference)  # candidate_list, given preference
    static_preference_index = cuda_(static_preference_index)
    static_score, _, _ = FM_model(mini_ui_pair, None, static_preference_index)
    static_score = static_score.detach().cpu().numpy()

    assert len(candidate_list) == len(static_score)
    assert candidate_list[-1] == itemID
    target_value = static_score[-1]
    #target_index = candidate_list.index(itemID)
    #target_value = static_score[target_index]
    ranked_score = nlargest(100000, static_score)
    target_position = ranked_score.index(target_value)
    print('Target position is: {} in {}'.format(target_position, len(ranked_score)))

    predictions = static_score.reshape((len(static_score), 1)[0])
    y_true = [0] * len(predictions)
    y_true[-1] = 1
    tmp = list(zip(y_true, predictions))
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    y_true, predictions = zip(*tmp)
    if len(y_true) > 1:
        auc = roc_auc_score(y_true, predictions)
        #print('auc is: {}'.format(auc))
        with open(write_fp, 'a') as f:
            f.write('Target position is: {} in {}, AUC: {}\n'.format(target_position, len(ranked_score), auc))

    print('FM_inference takes: {} seconds'.format(time.time() - start))
    return None

    #a = torch.LongTensor(residual_feature)
    #test_preference_matrix = a.expand(len(candidate_list), len(residual_feature))
    #test_preference_matrix = cuda_(test_preference_matrix)
#
    #for index, test_preference in enumerate(residual_feature):
    #    mini_preference_index = torch.cat([static_preference_index, test_preference_matrix[:, index].view(-1, 1)], dim=1)
    #    test_score, _, _ = FM_model(mini_ui_pair, None, mini_preference_index)
    #    test_score = test_score.detach().cpu().numpy()
    #    small_feature_raw_result_from_FM[test_preference] = test_score
    #
    ##print(len(residual_feature), len(small_feature_raw_result_from_FM))
    ##print(small_feature_raw_result_from_FM[test_preference])
#
    #start = time.time()
    #TopK = 50
    #change_dict = dict()
    #for k, v in small_feature_raw_result_from_FM.items():
    #    change_dict[k] = evaluate_change(static_score.reshape(-1).tolist(), v.reshape(-1).tolist(), TopK)
#
    #change_dict_big = defaultdict(list)
#
    #for k, v in big_small_mapping.items():
    #    for item in v:
    #        change_dict_big[k].append(change_dict[item])
#
    #TopN = 3
    #result_dict = dict()
    #for k, v in change_dict_big.items():
    #    result_dict[k] = sum(nsmallest(TopN, v))
#
    ## TODO: Cautious here, it means the smaller, the better
    #print('sorting takes: {} seconds'.format(time.time() - start))
#
    #return result_dict
