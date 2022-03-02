# BB-8 and R2-D2 are best friends.

import sys

sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

# from FM_model import *
import itertools
import json
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
import time
import torch
from FM_old import FactorizationMachine
from pn import PolicyNetwork
import random
import sampler
from redefine_popularity_distribution import global_data as data

emb_size = 64


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class _Config():
    def __init__(self):
        self.init_basic()
        self.init_type()

        self.init_misc()
        self.init_test()
        self.init_FM_related()
        self.init_bandit_parm()

    def init_basic(self):
        with open('../../data/FM-train-data/review_dict_train.json', 'r') as f:
            self._train_user_to_items = json.load(f)
        with open('../../data/FM-train-data/review_dict_valid.json', 'r') as f:
            self._valid_user_to_items = json.load(f)
        with open('../../data/FM-train-data/review_dict_test.json', 'r') as f:
            self._test_user_to_items = json.load(f)
        with open('../../data/FM-train-data/FM_busi_list.pickle', 'rb') as f:
            self.busi_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_user_list.pickle', 'rb') as f:
            self.user_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_train_list.pickle', 'rb') as f:
            self.train_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_valid_list.pickle', 'rb') as f:
            self.valid_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_test_list.pickle', 'rb') as f:
            self.test_list = pickle.load(f)

        # _______ item info _______
        with open('../../data/FM-train-data/item_dict-merge.json', 'r') as f:
            self.item_dict = json.load(f)

        with open('../../data/FM-train-data/test_photo_record.json', 'r') as f:
            self.test_photo_record = json.load(f)

        with open('../../data/FM-train-data/busi_list_test.json', 'r') as f:
            busi_list = json.load(f)
        with open('../../data/FM-train-data/photo_trans.json', 'r') as f:
            photo_trans = json.load(f)
        self.busi_list_test = [photo_trans[str(photo)] for photo in busi_list]

        with open('../../data/FM-train-data/redefined_item_popularity.json', 'r') as f:
            item_dict = json.load(f)
        temp_list = sorted(item_dict.items(), key=lambda dict: dict[1], reverse=True)
        self.popular_items = [int(a_0[0]) for a_0 in temp_list]

        with open('../../data/FM-train-data/item_positive_distribution.json', 'r') as f:
            item_dict = json.load(f)
        temp_list = sorted(item_dict.items(), key=lambda dict: dict[1], reverse=True)
        self.positive_items = [int(a_0[0]) for a_0 in temp_list]

    def init_type(self):
        self.INFORM_FACET = 'INFORM_FACET'
        self.ACCEPT_REC = 'ACCEPT_REC'
        self.REJECT_REC = 'REJECT_REC'

        # define agent behavior
        self.ASK_FACET = 'ASK_FACET'
        self.MAKE_REC = 'MAKE_REC'
        self.FINISH_REC_ACP = 'FINISH_REC_ACP'
        self.FINISH_REC_REJ = 'FINISH_REC_REJ'
        self.EPISODE_START = 'EPISODE_START'

        # define the sender type
        self.USER = 'USER'
        self.AGENT = 'AGENT'

    def init_misc(self):

        self.FACET_POOL = [i for i in range(31)]

        print('Total feature length is: {}, Top 30 namely: {}'.format(len(self.FACET_POOL), self.FACET_POOL[: 30]))
        self.REC_NUM = 10
        self.MAX_TURN = 15
        self.play_by = None
        self.calculate_all = None
        self.turn_count = np.zeros((16, 1))
        self.turn_count = np.zeros((16, 1))
        self.ask_count = np.zeros((16, 1))
        self.itm_psi = np.zeros((16, 1))
        self.att_psi = np.zeros((16, 1))

    def init_FM_related(self):
        city_max = 0
        category_max = 13
        feature_max = 0
        feature_min = 30
        for k, v in self.item_dict.items():

            if max(v['feature_index']) > feature_max:
                feature_max = max(v['feature_index'])
            if min(v['feature_index']) < feature_min:
                feature_min = min(v['feature_index'])

        print('feature max = {}'.format(feature_max))
        print('feature min = {}'.format(feature_min))

        stars_list = [1, 2, 3, 4, 5]
        price_list = [1, 2, 3, 4]
        self.star_count, self.price_count = len(stars_list), len(price_list)
        self.city_count, self.category_count, self.feature_count = city_max + 1, category_max + 1, feature_max + 1

        self.city_span = (0, self.city_count)
        self.star_span = (self.city_count, self.city_count + self.star_count)
        self.price_span = (self.city_count + self.star_count, self.city_count + self.star_count + self.price_count)

        self.spans = [self.city_span, self.star_span, self.price_span]

        print('city max: {}, category max: {}, feature max: {}'.format(self.city_count, self.category_count,
                                                                       self.feature_count))
        fp = '../../data/FM-model-merge/KS_FM_model_2.pt'
        model = FactorizationMachine(emb_size=64, user_length=len(self.user_list), item_length=len(self.busi_list),
                                     feature_length=feature_max + 1, qonly=1, command=8, hs=64, ip=0.01,
                                     dr=0.5, old_new='new')  # TODO: change later
        model.load_state_dict(torch.load(fp, map_location='cpu'))
        print('load FM model {}, but hypyer parameters can be mistaken'.format(fp))
        self.emb_matrix = model.feature_emb.weight[..., :-1].detach().numpy()
        self.user_emb = model.ui_emb.weight[..., :-1].detach().numpy()
        # print('length_user_emb = {}'.format(len(self.user_emb)))

        self.FM_model = cuda_(model)

    def init_test(self):
        pass


    def init_bandit_parm(self):
        self.user_TS_matrix = [np.eye(emb_size, emb_size) for i in range(len(self.user_emb))]
        self.user_TS_matrix_inv = [np.eye(emb_size, emb_size) for i in range(len(self.user_emb))]
        self.user_TS_f = [self.user_emb[i].reshape(emb_size, 1) for i in range(len(self.user_emb))]

    def change_param(self, playby, eval, update_count, update_reg, purpose, mod):
        self.play_by = playby
        self.eval = eval
        self.update_count = update_count
        self.update_reg = update_reg
        self.purpose = purpose
        self.mod = mod

    def random_sample(self, alpha):

        self._test_user_to_items = dict()
        self.test_list = list()
        _, _, self.test_list = sampler.random_sampling(sample_ratio=alpha)

        for row in self.test_list:
            _, user, item = row
            if str(user) not in self._test_user_to_items:
                self._test_user_to_items[str(user)] = list()
            self._test_user_to_items[str(user)].append(int(item))

        random.shuffle(self.test_list)


    def ratio_sample(self, alpha):

        self._test_user_to_items = dict()
        self.test_list = list()
        _, _, self.test_list = sampler.popular_sampling(sample_ratio=alpha)

        for row in self.test_list:
            _, user, item = row
            if str(user) not in self._test_user_to_items:
                self._test_user_to_items[str(user)] = list()
            self._test_user_to_items[str(user)].append(int(item))

        random.shuffle(self.test_list)


    def select_sample(self, alpha):

        self._test_user_to_items = dict()
        self.test_list = list()
        _, _, self.test_list = sampler.select_sampling(sample_ratio=alpha)

        for row in self.test_list:
            _, user, item = row
            if str(user) not in self._test_user_to_items:
                self._test_user_to_items[str(user)] = list()
            self._test_user_to_items[str(user)].append(int(item))

        random.shuffle(self.test_list)



start = time.time()
global_config = _Config()
print('Config takes: {}'.format(time.time() - start))

print('___Config Done!!___')
