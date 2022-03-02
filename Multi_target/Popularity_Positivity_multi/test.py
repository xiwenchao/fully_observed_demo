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

with open('../../data/FM-train-data/redefined_item_popularity.json', 'r') as f:
    popular_item = json.load(f)


with open('../../data/FM-train-data/item_positive_distribution.json', 'r') as f:
    positive_item = json.load(f)


with open('../../data/FM-train-data/item_dict-merge.json', 'r') as f:
    item_dict_att = json.load(f)

tem_dict = dict()
for i in range(31):
    tem_dict[i] = 0
for item in popular_item.keys():
    for attribute in item_dict_att[str(item)]["feature_index"]:
        tem_dict[int(attribute)] += popular_item[item]

temp_list = sorted(tem_dict.items(), key=lambda dict: dict[1], reverse=True)
popular_attributes = [int(a_0[0]) for a_0 in temp_list]

print(popular_attributes)

tem_dict = dict()
for i in range(31):
    tem_dict[i] = 0
for item in positive_item.keys():
    for attribute in item_dict_att[str(item)]["feature_index"]:
        tem_dict[int(attribute)] += positive_item[item]

temp_list = sorted(tem_dict.items(), key=lambda dict: dict[1], reverse=True)
positive_attributes = [int(a_0[0]) for a_0 in temp_list]

print(positive_attributes)