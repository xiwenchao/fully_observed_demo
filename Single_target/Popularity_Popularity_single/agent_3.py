# BB-8 and R2-D2 are best friends.

import sys
import time
from collections import defaultdict
import random

random.seed(0)
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.distributions import Categorical

from message import message
from config_3 import global_config as cfg
from utils_entropy import cal_ent
from heapq import nlargest, nsmallest
from utils_fea_sim_3 import feature_similarity
from utils_fea_sim_3 import feature_similarity_micro
from utils_sense_3 import try_feature_cause_change, rank_items
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import math

d = 64


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class agent():
    def __init__(self, FM_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo,
                 PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini,
                 optimizer1_fm, optimizer2_fm, alwaysupdate, epsilon, sample_dict, choose_pool):
        # _______ input parameters_______
        self.user_id = user_id
        self.busi_id = busi_id
        self.FM_model = FM_model

        self.turn_count = 0
        self.F_dict = defaultdict(lambda: defaultdict())
        self.recent_candidate_list = cfg.popular_items
        self.recent_candidate_list_ranked = self.recent_candidate_list  # TODO: We only initialize this way

        self.asked_feature = list()
        self.do_random = do_random
        self.rejected_item_list_ = list()

        self.history_list = list()

        self.write_fp = write_fp
        self.strategy = strategy
        self.TopKTaxo = TopKTaxo
        self.entropy_dict_10 = None
        self.entropy_dict_50 = None
        self.entropy_dict = None
        self.sim_dict = None
        self.sim_dict2 = None
        self.PN_model = PN_model

        self.known_feature = list()
        self.known_facet = list()

        self.residual_feature_big = None
        self.skip_big_feature = list()


        self.log_prob_list = log_prob_list
        self.action_tracker = action_tracker
        self.candidate_length_tracker = candidate_length_tracker
        self.mini_update_already = False
        self.mini = mini
        self.optimizer1_fm = optimizer1_fm
        self.optimizer2_fm = optimizer2_fm
        self.alwaysupdate = alwaysupdate
        self.previous_dict = None
        self.rejected_time = 0
        self.big_feature_length = 31
        self.feature_length = 31
        self.sample_dict = sample_dict
        self.choose_pool = choose_pool

    def get_batch_data(self, pos_neg_pairs, bs, iter_):
        PAD_IDX1 = len(cfg.user_list) + len(cfg.item_dict)
        PAD_IDX2 = cfg.feature_count

        left = iter_ * bs  # bs: batch size
        right = min((iter_ + 1) * bs, len(pos_neg_pairs))
        pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
        for instance in pos_neg_pairs[left: right]:
            # instance[0]: pos item, instance[1] neg item
            pos_list.append(torch.LongTensor([self.user_id, instance[0] + len(cfg.user_list)]))
            f = cfg.item_dict[str(instance[0])]['feature_index']
            # f = [PAD_IDX2]
            pos_list2.append(torch.LongTensor(f))

            neg_list.append(torch.LongTensor([self.user_id, instance[1] + len(cfg.user_list)]))
            f = cfg.item_dict[str(instance[1])]['feature_index']
            # f = [PAD_IDX2]
            neg_list2.append(torch.LongTensor(f))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)

        neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
        neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)


        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)


    def mini_update_FM(self, count):
        self.FM_model.train()
        bs = 32
        if str(self.user_id) in cfg._train_user_to_items:
            pos_items = cfg._train_user_to_items[str(self.user_id)]
        else:
            pos_items = cfg._test_user_to_items[str(self.user_id)]

        neg_items = self.rejected_item_list_[-10:]

        known_items = list()
        if str(self.user_id) in cfg._train_user_to_items:
            known_items += cfg._train_user_to_items[str(self.user_id)]
        if str(self.user_id) in cfg._valid_user_to_items:
            known_items += cfg._valid_user_to_items[str(self.user_id)]
        if str(self.user_id) in cfg._test_user_to_items:
            known_items += cfg._test_user_to_items[str(self.user_id)]

        random_neg = list(set(cfg.busi_list_test) - set(known_items))
        if cfg.mix_pos == 1:
            pos_items = pos_items + random.sample(random_neg, len(pos_items))  # to avoid overfitting
        if cfg.mix_neg == 1:
            neg_items = neg_items + random.sample(random_neg, len(neg_items))

        pos_neg_pairs = list()
        for p_item in pos_items:
            for n_item in neg_items:
                pos_neg_pairs.append((p_item, n_item))

        pos_neg_pairs = list()

        num = int(bs / len(pos_items)) + 1
        pos_items = pos_items * num

        for p_item in pos_items:
            n_item = random.choice(neg_items)
            pos_neg_pairs.append((p_item, n_item))
        random.shuffle(pos_neg_pairs)

        max_iter = int(len(pos_neg_pairs) / bs)

        reg_ = torch.Tensor([cfg.update_reg])
        reg_ = torch.autograd.Variable(reg_, requires_grad=False)
        reg_ = cuda_(reg_)
        reg = reg_

        lsigmoid = nn.LogSigmoid()
        for iter_ in range(max_iter):
            pos_list, pos_list2, neg_list, neg_list2 = self.get_batch_data(pos_neg_pairs, bs, iter_)
            result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = self.FM_model(pos_list, None, pos_list2)
            result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = self.FM_model(neg_list, None, neg_list2)
            # result_pos = cuda_(torch.zeros(bs, 1))
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)

            nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            loss += (reg * nonzero_matrix_pos_).sum(dim=0)
            loss += (reg * nonzero_matrix_neg_).sum(dim=0)

            self.optimizer2_fm.zero_grad()
            loss.backward()
            self.optimizer2_fm.step()



    def vectorize(self):

        list3 = [v for k, v in self.entropy_dict.items()]
        list4 = [v for k, v in self.sim_dict2.items()]

        list5 = self.history_list[:15] + [0] * (15 - len(self.history_list[:15]))

        list6 = [0] * 8
        if len(self.recent_candidate_list) <= 10:
            list6[0] = 1
        if len(self.recent_candidate_list) > 10 and len(self.recent_candidate_list) <= 50:
            list6[1] = 1
        if len(self.recent_candidate_list) > 50 and len(self.recent_candidate_list) <= 100:
            list6[2] = 1
        if len(self.recent_candidate_list) > 100 and len(self.recent_candidate_list) <= 200:
            list6[3] = 1
        if len(self.recent_candidate_list) > 200 and len(self.recent_candidate_list) <= 300:
            list6[4] = 1
        if len(self.recent_candidate_list) > 300 and len(self.recent_candidate_list) <= 500:
            list6[5] = 1
        if len(self.recent_candidate_list) > 500 and len(self.recent_candidate_list) <= 1000:
            list6[6] = 1
        if len(self.recent_candidate_list) > 1000:
            list6[7] = 1


        list_cat = list3 + list4 + list5 + list6

        list_cat = np.array(list_cat)

        return list_cat


    def vectorize_crm(self):
        a = [0] * self.feature_length
        for item in self.known_feature:
            a[item] = 1
        return np.array(a)


    def update_upon_feature_inform(self, input_message):
        assert input_message.message_type == cfg.INFORM_FACET

        # _______ update F_dict________
        facet = input_message.data['facet']
        if facet is None:
            print('?')
        self.asked_feature.append(facet)

        value = input_message.data['value']

        if value is not None:
                self.recent_candidate_list = [k for k in self.recent_candidate_list if
                                              set(value).issubset(set(cfg.item_dict[str(k)]['feature_index']))]
                self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [
                    self.busi_id]

                if type(value) != int:
                    self.known_feature += [int(i) for i in
                                           value]  # Update known feature # TODO: we have bug here in last version
                else:
                    self.known_feature += [int(value)]

                self.known_feature = list(set(self.known_feature))
                self.known_facet.append(facet)

                # dictionary
                l = list(set(self.recent_candidate_list) - set([self.busi_id]))
                random.shuffle(l)
                if cfg.play_by == 'AOO':
                    self.sample_dict[self.busi_id].append((self.known_feature, l[: 10]))

                if cfg.play_by != 'AOO':
                    self.sim_dict = feature_similarity(self.known_feature, self.user_id, self.TopKTaxo)
                    self.sim_dict2 = self.sim_dict.copy()

        if (value is not None and value[0] is not None) or self.turn_count == 1:
            c = cal_ent(self.recent_candidate_list[: 10])
            d = c.do_job()
            self.entropy_dict_10 = d
            c = cal_ent(self.recent_candidate_list[: 50])
            d = c.do_job()
            self.entropy_dict_50 = d

            c = cal_ent(self.recent_candidate_list)
            d = c.do_job()
            self.entropy_dict = d

        for f in self.asked_feature:
            self.entropy_dict[f] = 0

        for f in self.asked_feature:
            if self.sim_dict2 is not None and f in self.sim_dict:
                self.sim_dict2[f] = -1
                if self.entropy_dict != None:
                    if self.entropy_dict[f] == 0:
                        self.sim_dict[f] = -1

        self.residual_feature_big = list(set(self.choose_pool) - set(self.known_facet))

    def prepare_next_question(self):
        if self.strategy == 'maxent':
            facet = max(self.entropy_dict, key=self.entropy_dict.get)
            data = dict()
            data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        elif self.strategy == 'maxsim':
            for f in self.asked_feature:
                if self.sim_dict is not None and f in self.sim_dict:
                    self.sim_dict[f] = -1
            # facet = max(self.sim_dict, key=self.sim_dict.get)
            if len(self.known_feature) == 0 and self.sim_dict is None:
                facet = max(self.entropy_dict, key=self.entropy_dict.get)
            else:
                facet = max(self.sim_dict, key=self.sim_dict.get)
            data = dict()
            data['facet'] = facet

            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            # print('ask facet {}'.format(facet))
            return new_message
        else:
            pool = [item for item in cfg.FACET_POOL if item not in self.asked_feature]
            facet = np.random.choice(np.array(pool), 1)[0]
            data = dict()
            if facet in [item.name for item in cfg.cat_tree.children]:
                data['facet'] = facet
            else:
                data['facet'] = facet

            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            return new_message

    def prepare_rec_message(self):
        self.recent_candidate_list_ranked = [item for item in self.recent_candidate_list_ranked if
                                             item not in self.rejected_item_list_]  # Delete those has been rejected
        rec_list = self.recent_candidate_list_ranked[: 10]
        data = dict()
        data['rec_list'] = rec_list
        new_message = message(cfg.AGENT, cfg.USER, cfg.MAKE_REC, data)
        return new_message

    def response(self, input_message):

        assert input_message.sender == cfg.USER
        assert input_message.receiver == cfg.AGENT

        # _______ update the agent self_______
        if input_message.message_type == cfg.INFORM_FACET:

            if input_message.data['value'] is None:
                self.history_list.append(0)

            else:
                self.history_list.append(1)

            self.update_upon_feature_inform(input_message)

        if input_message.message_type == cfg.REJECT_REC:
            self.rejected_item_list_ += input_message.data['rejected_item_list']
            self.rejected_time += 1

        if input_message.message_type == cfg.REJECT_REC:
            self.history_list.append(-1)
        action = None
        SoftMax = nn.Softmax()
        if cfg.play_by == 'AOO' or cfg.play_by == 'AOO_valid':
            new_message = self.prepare_next_question()  #

        if cfg.play_by == 'AO':  # means AskOnly
            action = 0
            new_message = self.prepare_next_question()

            if cfg.hardrec == 'two':
                x = len(self.recent_candidate_list)
                p = 10.0 / x
                a = random.uniform(0, 1)
                if a < p:
                    new_message = self.prepare_rec_message()

        if cfg.play_by == 'Naive':  # means AskOnly
            action = 0
            new_message = self.prepare_next_question()

            a = random.uniform(0, 1)
            if a > 0.5:
                new_message = self.prepare_rec_message()

        if cfg.play_by == 'RO':  # means RecOnly
            new_message = self.prepare_rec_message()
        if cfg.play_by == 'AR':  # means Ask and Recommend
            action = random.randint(0, 1)


        if cfg.play_by == 'policy':  # do policy gradient


            if cfg.eval == 1:
                action_max = 31

                action = Variable(torch.IntTensor([action_max]))

            if action < 31:
                # print('ask facet')
                data = dict()
                data['facet'] = cfg.FACET_POOL[action]
                new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            else:
                # print('recommend')
                new_message = self.prepare_rec_message()

        if cfg.play_by == 'policy':
            self.action_tracker.append(action.data.numpy().tolist())

            self.candidate_length_tracker.append(len(self.recent_candidate_list))


        new_message.data['itemID'] = self.busi_id
        return new_message

