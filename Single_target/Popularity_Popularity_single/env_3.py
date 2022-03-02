# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

from collections import Counter
import numpy as np
from random import randint
import json
import random

from message import message
from config_3 import global_config as cfg
import time


class user():
    def __init__(self, user_id, busi_id, uc):
        self.user_id = user_id
        self.busi_id = busi_id
        self.uc = uc  # means uncertainty
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]

    def find_brother(self, node):
        return [child.name for child in node.parent.children if child.name != node.name]

    def find_children(self, node):
        return [child.name for child in node.children if child.name != node.name]

    def inform_facet(self, facet):
        data = dict()
        data['facet'] = facet
        if facet in ['stars']:
            rand_list = np.random.choice(np.arange(0, 2), 10, p=[self.uc, 1 - self.uc]).tolist()
            if rand_list[0] == 0:
                data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
            if rand_list[0] > 0:
                data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['RestaurantsPriceRange2']:
            rand_list = np.random.choice(np.arange(0, 2), 10, p=[self.uc, 1 - self.uc]).tolist()
            if rand_list[0] == 0:
                data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
            if rand_list[0] > 0:
                data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
                if cfg.item_dict[str(self.busi_id)][facet] is None:
                    data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['city']:
            rand_list = np.random.choice(np.arange(0, 2), 10, p=[self.uc, 1 - self.uc]).tolist()
            if rand_list[0] == 0:
                data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
            if rand_list[0] > 0:
                data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        else:  # means now deal with big features

            candidate_feature = [facet]
            ground_truth_feature = cfg.item_dict[str(self.busi_id)]['feature_index']
            intersection_between = list(set(candidate_feature).intersection(set(ground_truth_feature)))

            if len(intersection_between) == 0:
                data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

            data['value'] = intersection_between
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

    def inform_facet_demo(self, facet):
        data = dict()
        data['facet'] = facet
        if facet in ['stars']:
            print('We want to ask you about star')
            data['value'] = [int(input('Star Ratings: 1, 2, 3, 4, 5, which do you like?    '))]
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['RestaurantsPriceRange2']:
            print('We want to ask you about price')
            data['value'] = [float(input('Price Ranges: 1, 2, 3, 4, which do you like?    '))]
            # data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['city']:
            l = ''
            for k, v in cfg.city_map.items():
                l += '{}:{}  '.format(k, v)
            l = l.encode('utf-8').strip()
            print(l)
            data['value'] = [int(input('We have above cities, which are you in?    '))]

        else:  # means now deal with big features
            # TODO: add dropout here!
            candidate_feature = cfg.taxo_dict[facet]
            for id in candidate_feature:
                print(id, cfg.tag_map_inverted[id])
            print('For "{}" category, we have above attributes'.format(facet))
            answer = input('Which ones do you like? Separate by comma:   ')
            data['value'] = [int(item) for item in answer.split(',')]
            print('Got you!\n')

        return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

    def response(self, input_message):
        assert input_message.sender == cfg.AGENT
        assert input_message.receiver == cfg.USER

        # _______ update candidate _______
        if 'candidate' in input_message.data: self.recent_candidate_list = input_message.data['candidate']

        new_message = None

        if input_message.message_type == cfg.ASK_FACET:
            facet = input_message.data['facet']
            new_message = self.inform_facet(facet)

        if input_message.message_type == cfg.MAKE_REC or input_message.message_type == cfg.EPISODE_START:
            # ----- demo use begin -----
            rec_list = input_message.data['rec_list']
            rec_list = input_message.data['rec_list']

            if input_message.data['itemID'] not in rec_list:
                user_feedback = 'no'
                data = dict()
                data['rejected_item_list'] = input_message.data['rec_list']
                new_message = message(cfg.USER, cfg.AGENT, cfg.REJECT_REC, data)

            else:
                user_feedback = 'yes'
                data = dict()
                data['ranking'] = 1
                data['total'] = 1
                new_message = message(cfg.USER, cfg.AGENT, cfg.ACCEPT_REC, data)
                print("Got you! You like these restaurants, let's move on!")

            if user_feedback == 'quit':
                new_message = message(cfg.USER, cfg.AGENT, 'quit', data)


            new_message.data['itemID'] = input_message.data['itemID']
            # ----- demo use end -----

        return new_message
