# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 4:17 下午
# @Author  : Chongming GAO
# @FileName: matrix_data.py
import itertools
import os
import random

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from util import Dataset
import util


class ID_Translator:
    def __init__(self, item_id_user_bigID, item_id_small_bigID):
        self.le_user = LabelEncoder()
        self.le_item = LabelEncoder()

        self.le_user.fit(item_id_user_bigID)
        self.le_item.fit(item_id_small_bigID)

        self.num_user = len(item_id_user_bigID)
        self.num_item = len(item_id_small_bigID)


class Matrix_data:
    def __init__(self, user_positive_bigID, num_user, num_item, translator=None):

        if not translator is None:
            user_positive_item = {
                translator.le_user.transform([int(k)])[0]:
                    translator.le_item.transform(v)
                for k, v in user_positive_bigID.items()}
        else:
            user_positive_item = user_positive_bigID

        self.user_positive_item = user_positive_item
        self.num_user = num_user
        self.num_item = num_item
        self.translator = translator

        cols = np.concatenate(list(user_positive_item.values()))[:]
        rows = np.concatenate([len(v) * [k] for k,v in user_positive_item.items()])
        values = np.ones([len(rows)])

        self.matrix = csr_matrix((values, (rows, cols)), dtype=np.int16, shape=(self.num_user, self.num_item))


if __name__ == '__main__':
    data = Dataset()
    data.user_positive_bigID = data.load_file('review_dict_test.json')
    # data.item_feature_bigID = data.load_file('item_dict-merge.json')
    # data.item_exposure_distribution_bigID = data.load_file('item_exposure_distribution.json')
    # data.item_positive_distribution_bigID = data.load_file('item_positive_distribution.json')

    item_id_small_rawID = data.load_file('busi_list_test.json')
    # Dictionary from raw id to id.
    data.photo_trans_from_rawID_to_bigID = data.load_file("photo_trans.json")
    # small data's ID in [0, 10728]
    item_id_small_bigID = [data.photo_trans_from_rawID_to_bigID[str(photo)] for photo in item_id_small_rawID]

    user_id_small_bigID = [int(i) for i in data.user_positive_bigID.keys()]

    ID_small_big_translator = ID_Translator(user_id_small_bigID, item_id_small_bigID)


    matrix = Matrix_data(data.user_positive_bigID, ID_small_big_translator.num_user, ID_small_big_translator.num_item, translator=ID_small_big_translator)


