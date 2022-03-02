# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import pickle
import torch
import argparse

import time
import numpy as np
import json

from config_3 import global_config as cfg
from FM_old import FactorizationMachine
from epi_3 import run_one_episode, update_PN_model
from pn import PolicyNetwork
from FM_old import FactorizationMachine
import copy

from collections import defaultdict

import random
import os.path
import json
import math
import regex as re

# import torchvision.models as models



random.seed(1)

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print(the_max)
FEATURE_COUNT = the_max + 1


def cuda_(var):
    return var.cuda() if torch.cuda.is_available()else var


def main():
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN')
    parser.add_argument('-playby', type=str, dest='playby', help='playby')
    parser.add_argument('-fmCommand', type=str, dest='fmCommand', help='fmCommand')
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer')
    parser.add_argument('-lr', type=float, dest='lr', help='lr')
    parser.add_argument('-decay', type=float, dest='decay', help='decay')
    parser.add_argument('-TopKTaxo', type=int, dest='TopKTaxo', help='TopKTaxo')
    parser.add_argument('-gamma', type=float, dest='gamma', help='gamma')
    parser.add_argument('-trick', type=int, dest='trick', help='trick')

    parser.add_argument('-startFrom', type=int, dest='startFrom', help='startFrom')
    parser.add_argument('-endAt', type=int, dest='endAt', help='endAt')
    parser.add_argument('-strategy', type=str, dest='strategy', help='strategy')
    parser.add_argument('-eval', type=int, dest='eval', help='eval')
    parser.add_argument('-mini', type=int, dest='mini', help='mini')  # means mini-batch update the FM
    parser.add_argument('-alwaysupdate', type=int, dest='alwaysupdate',
                        help='alwaysupdate')  # means mini-batch update the FM
    parser.add_argument('-initeval', type=int, dest='initeval', help='initeval')
    parser.add_argument('-upoptim', type=str, dest='upoptim', help='upoptim')
    parser.add_argument('-uplr', type=float, dest='uplr', help='uplr')
    parser.add_argument('-upcount', type=int, dest='upcount', help='upcount')
    parser.add_argument('-upreg', type=float, dest='upreg', help='upreg')
    parser.add_argument('-code', type=float, dest='code', help='code')
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose')  # options: pretrain, fmdata, others
    parser.add_argument('-mod', type=str, dest='mod', help='mod')  # options: CRM, EAR
    parser.add_argument('-alpha', type=float, dest='alpha', help='alpha')

    A = parser.parse_args()

    # Note:
    # purpose = fmdata, playby: AOO, AOO_valid, are for sample training data and validation data.


    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for i in range(10):

            cfg.ratio_sample(alpha=alpha)
            cfg.init_misc()
            cfg.init_FM_related()
            cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg,
                             purpose=A.purpose, mod=A.mod)

            random.seed(1)
            random.shuffle(cfg.valid_list)
            random.shuffle(cfg.test_list)

            the_valid_list = copy.copy(cfg.valid_list)
            the_test_list = copy.copy(cfg.test_list)
            random.shuffle(the_valid_list)
            random.shuffle(the_test_list)

            print('valid length: {}, test list length: {}'.format(len(the_valid_list), len(the_test_list)))
            # sys.sleep(1)

            gamma = A.gamma
            FM_model = cfg.FM_model


            INPUT_DIM = 0
            if A.mod == 'ear':
                INPUT_DIM = 85
            if A.mod == 'crm':
                INPUT_DIM = 590

            PN_model = 'PolicyNetwork'
            start = time.time()
            end_point = A.endAt

            sample_dict = defaultdict(list)

            total_turn = 0
            epi_final = 0
            for epi_count in range(A.startFrom, A.endAt):
                if epi_count % 100 == 0:
                    print('It has processed {} episodes'.format(epi_count))
                if A.eval == 1:
                    if epi_count >= len(cfg.test_list):
                        continue

                epi_final += 1

                current_FM_model = copy.deepcopy(FM_model)
                cuda_(current_FM_model)
                param1, param2 = list(), list()
                param3 = list()
                i = 0
                for name, param in current_FM_model.named_parameters():
                    # print(name, param)
                    if i == 0:
                        param1.append(param)
                    else:
                        param2.append(param)
                    if i == 2:
                        param3.append(param)
                        param.requires_grad = False
                    i += 1

                optimizer1_fm, optimizer2_fm = None, None
                if A.purpose != 'fmdata':
                    optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01,
                                                        weight_decay=A.decay)  # TODO: change learning rate
                    if A.upoptim == 'Ada':
                        optimizer2_fm = torch.optim.Adagrad(param2, lr=A.uplr, weight_decay=A.decay)
                    if A.upoptim == 'SGD':
                        optimizer2_fm = torch.optim.SGD(param2, lr=0.001, weight_decay=A.decay)
                # end following

                if A.purpose != 'fmdata':
                    _, u, item = cfg.test_list[epi_count]
                    user_id = int(u)
                    item_id = int(item)
                else:
                    user_id = 0
                    item_id = epi_count

                print('\n\n\nHello! I am glad to serve you!')

                write_fp = '../../data/interaction-log/{}/v6-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}.txt'.format(
                    A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo,
                    A.trick,
                    A.eval, A.initeval,
                    A.mini, A.alwaysupdate, A.upcount, A.upreg)

                choose_pool = cfg.item_dict[str(item_id)]['feature_index']

                choose_pool_original = choose_pool
                if A.purpose not in ['pretrain', 'fmdata']:
                    choose_pool = [random.choice(choose_pool)]

                for c in choose_pool:
                    start_facet = c

                    if A.purpose != 'pretrain':
                        log_prob_list, rewards, turn_count = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet, sample_dict, choose_pool_original)
                    else:
                        run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                        A.strategy, A.TopKTaxo,
                                        PN_model, gamma, A.trick, A.mini,
                                        optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet, sample_dict,
                                        choose_pool_original)
                # end run
                total_turn += turn_count


            print('average_turn = {}'.format(total_turn / (epi_final)))
            print('successful recommend rate = {}'.format(cfg.turn_count / (epi_final)))

            SR_15, SR_10, SR_5 = 0, 0, 0
            k = 0
            for row in cfg.turn_count:
                SR_15 += row[0]
                if k < 6:
                    SR_5 += row[0]
                if k < 11:
                    SR_10 += row[0]
                k += 1
            print('SR@15 = {}'.format(SR_15 / (epi_final)))

            # f.write('successful recommend rate = {} \n'.format(cfg.turn_count / (epi_final)))
            with open("final_results_{}.txt".format(alpha), 'a') as f:
                f.write('average_turn = {} \n'.format(total_turn / (epi_final)))

                f.write('SR@15 = {} \n'.format(SR_15 / (epi_final)))
                f.write('SR@10 = {} \n'.format(SR_10 / (epi_final)))
                f.write('SR@5 = {} \n'.format(SR_5 / (epi_final)))
                f.write('\n \n')


if __name__ == '__main__':
    main()