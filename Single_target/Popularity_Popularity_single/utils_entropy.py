from collections import Counter
import numpy as np
from config_3 import global_config as cfg
import time


class cal_ent():
    '''
    given the current candidate list, calculate the entropy of every feature(denote by f)
    '''

    def __init__(self, recent_candidate_list):
        self.recent_candidate_list = recent_candidate_list

    def calculate_entropy_for_one_tag(self, tagID, _counter):
        '''
        Args:
        tagID: int
        '''
        v = _counter[tagID]
        p1 = float(v) / len(self.recent_candidate_list)
        p2 = 1.0 - p1

        if p1 == 0 or p1 == 1:
            return 0
        return (- p1 * np.log2(p1) - p2 * np.log2(p2))

    def do_job(self):
        entropy_dict_small_feature = dict()
        cat_list_all = list()
        for k in self.recent_candidate_list:
            cat_list_all += cfg.item_dict[str(k)]['feature_index']

        c = Counter(cat_list_all)
        #print('c is: {} (doing entropy calculation)'.format(len(self.recent_candidate_list)))
        for k, v in c.items():
            node_entropy_self = self.calculate_entropy_for_one_tag(k, c)
            # entropy_dict[cfg.tag_map_inverted[k]] = node_entropy_self
            entropy_dict_small_feature[k] = node_entropy_self
            # print('entropy_dict_small_feature[{}] = {}'.format(k, entropy_dict_small_feature[k]))
        for label in cfg.FACET_POOL:
            if label not in entropy_dict_small_feature:
                entropy_dict_small_feature[label] = 0

        entropy_dict = dict()
        entropy_dict = entropy_dict_small_feature
        '''
        for big_feature in cfg.FACET_POOL:  # means we deal with big features (starting from 3rd index)
            
            remained_small = [f for f in cfg.taxo_dict[big_feature] if f in entropy_dict_small_feature.keys()]
            if len(remained_small) == 0:
                entropy_dict[big_feature] = 0
                continue
            
            
        '''
        return entropy_dict

    def do_job_big(self):
        pass
