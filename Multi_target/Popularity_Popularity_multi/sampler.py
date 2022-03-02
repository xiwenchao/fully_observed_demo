# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 2:46 下午
# @Author  : Chongming GAO
# @FileName: sampler.py


import random
from collections import defaultdict
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

from matrix_data import Matrix_data, ID_Translator
from util import Dataset


class Sampler:
    def __init__(self, original_matrix):
        self.original_matrix = original_matrix
        self.num_user = self.original_matrix.num_user
        self.num_item = self.original_matrix.num_item
        self.all_num = self.num_user * self.num_item

    def int2index(self, integer):
        row = integer // self.num_item
        col = integer % self.num_item
        return np.array([[row], [col]])

    def int2row(self, integer):
        row = integer // self.num_item
        # col = integer % self.num_item
        return row

    def int2col(self, integer):
        # row = integer // self.num_item
        col = integer % self.num_item
        return col

    def get_sampled_matrix_from_sampled_integers(self, sampled_num):
        sampled_row = [self.int2row(i) for i in sampled_num]
        sampled_col = [self.int2col(i) for i in sampled_num]
        values = np.ones([len(sampled_row)])
        sampled_matrix = csr_matrix((values, (sampled_row, sampled_col)), dtype=np.int16,
                                    shape=(self.num_user, self.num_item))

        sampled_results = sampled_matrix.multiply(self.original_matrix.matrix)
        return sampled_results

    def get_result(self, sampled_results):
        # self._test_user_to_items = defaultdict(list)
        indices = sampled_results.indices
        test_user_to_items = {k: indices[l: r].tolist() for k, (l, r) in
                              enumerate(zip(sampled_results.indptr[:-1], sampled_results.indptr[1:]))}

        test_list_1 = list(chain(*[list(zip([k] * len(v), v)) for k, v in test_user_to_items.items()]))
        test_list = [(0, i, j) for (i, j) in test_list_1]

        if not self.original_matrix.translator is None:
            le_user = self.original_matrix.translator.le_user
            le_item = self.original_matrix.translator.le_item

            test_user_to_items_bigID = {str(le_user.inverse_transform([k])[0]): le_item.inverse_transform(v) for k, v in
                                        test_user_to_items.items()}

            test_list_bigID = list(zip([0] * len(test_list),
                                       le_user.inverse_transform([user for _, user, _ in test_list]),
                                       le_item.inverse_transform([item for _, _, item in test_list])))
            # [(0, user, item) for _, user, item in test_list]
            return sampled_results, test_user_to_items_bigID, test_list_bigID

        return sampled_results, test_user_to_items, test_list

    def random_sample(self, ratio):
        self._test_user_to_items = dict()
        self.test_list = list()

        sampled_num = random.sample(range(self.all_num), int(self.all_num * ratio))
        sampled_results = self.get_sampled_matrix_from_sampled_integers(sampled_num)

        return self.get_result(sampled_results)

    def biased_sample(self, ratio, item_probability):
        if not self.original_matrix.translator is None:
            le_user = self.original_matrix.translator.le_user
            le_item = self.original_matrix.translator.le_item
            item_probability_trans = {le_item.transform([int(k)])[0]: v for k, v in item_probability.items()}
            ordered_probability = {k: v for k, v in sorted(item_probability_trans.items(), key=lambda item: item[0])}
            item_probability = list(ordered_probability.values())
        probability = np.array(list(item_probability) * self.num_user)
        probability = probability / sum(probability)

        sampled_num = np.random.choice(self.all_num, size=int(ratio * self.all_num), replace=False, p=probability)
        sampled_results = self.get_sampled_matrix_from_sampled_integers(sampled_num)
        return self.get_result(sampled_results)


def random_sampling(sample_ratio=0.5):
    data = Dataset()
    data.user_positive_bigID = data.load_file('review_dict_test.json')

    item_id_small_rawID = data.load_file('busi_list_test.json')
    # Dictionary from raw id to id in Big matrix.
    data.photo_trans_from_rawID_to_bigID = data.load_file("photo_trans.json")
    # small data's ID in [0, 10728]
    item_id_small_bigID = [data.photo_trans_from_rawID_to_bigID[str(photo)] for photo in item_id_small_rawID]

    user_id_small_bigID = [int(i) for i in data.user_positive_bigID.keys()]

    ID_small_big_translator = ID_Translator(user_id_small_bigID, item_id_small_bigID)

    full_observed_matrix = Matrix_data(data.user_positive_bigID, ID_small_big_translator.num_user,
                                       ID_small_big_translator.num_item, translator=ID_small_big_translator)

    sampler = Sampler(full_observed_matrix)

    # For random sampling
    print("Sample Random items with ratio [{}]".format(sample_ratio))
    sampled_matrix, test_user_to_items, test_list = sampler.random_sample(sample_ratio)

    return sampled_matrix, test_user_to_items, test_list


def popular_sampling(sample_ratio=0.5):
    data = Dataset()
    data.user_positive_bigID = data.load_file('review_dict_test.json')

    item_id_small_rawID = data.load_file('busi_list_test.json')
    # Dictionary from raw id to id in Big matrix.
    data.photo_trans_from_rawID_to_bigID = data.load_file("photo_trans.json")
    # small data's ID in [0, 10728]
    item_id_small_bigID = [data.photo_trans_from_rawID_to_bigID[str(photo)] for photo in item_id_small_rawID]

    user_id_small_bigID = [int(i) for i in data.user_positive_bigID.keys()]

    ID_small_big_translator = ID_Translator(user_id_small_bigID, item_id_small_bigID)

    full_observed_matrix = Matrix_data(data.user_positive_bigID, ID_small_big_translator.num_user,
                                       ID_small_big_translator.num_item, translator=ID_small_big_translator)

    sampler = Sampler(full_observed_matrix)

    # For biased sampling
    data.defined_popular_distribution = data.load_file('redefined_item_popularity.json') # This file can be obtained by runing util.py
    data.item_positive_distribution = data.load_file('item_positive_distribution.json')

    print("Sample items with our defined popularity bias with sample ratio [{}]".format(sample_ratio))
    sampled_matrix, test_user_to_items, test_list = sampler.biased_sample(sample_ratio,
                                                                          data.defined_popular_distribution)

    return sampled_matrix, test_user_to_items, test_list


def select_sampling(sample_ratio=0.5):
    data = Dataset()
    data.user_positive_bigID = data.load_file('review_dict_test.json')

    item_id_small_rawID = data.load_file('busi_list_test.json')
    # Dictionary from raw id to id in Big matrix.
    data.photo_trans_from_rawID_to_bigID = data.load_file("photo_trans.json")
    # small data's ID in [0, 10728]
    item_id_small_bigID = [data.photo_trans_from_rawID_to_bigID[str(photo)] for photo in item_id_small_rawID]

    user_id_small_bigID = [int(i) for i in data.user_positive_bigID.keys()]

    ID_small_big_translator = ID_Translator(user_id_small_bigID, item_id_small_bigID)

    full_observed_matrix = Matrix_data(data.user_positive_bigID, ID_small_big_translator.num_user,
                                       ID_small_big_translator.num_item, translator=ID_small_big_translator)

    sampler = Sampler(full_observed_matrix)

    # For random sampling

    data.defined_popular_distribution = data.load_file('redefined_item_popularity.json') # This file can be obtained by runing util.py
    data.item_positive_distribution = data.load_file('item_positive_distribution.json')

    print("Sample items with item selection bias with sample ratio [{}]".format(sample_ratio))
    sampled_matrix, test_user_to_items, test_list = sampler.biased_sample(sample_ratio,
                                                                          data.item_positive_distribution)
    return sampled_matrix, test_user_to_items, test_list


if __name__ == '__main__':
    random_sampling()

    # matrix = Matrix_data({0: [0, 2], 1: [2], 2: [0, 1, 2]}, 3, 3)
    # sampler = Sampler(matrix)
    # a, aa = sampler.random_sample(1)
    # print(a, aa)
