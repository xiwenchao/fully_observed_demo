import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck

import sys
sys.path.insert(0, '../lib/actor-critic-2')

#from config import global_config as cfg

#random.seed(999)

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class FactorizationMachine(nn.Module):
    def __init__(self, emb_size, user_length, item_length, feature_length, qonly, command, hs, ip, dr, old_new):
        super(FactorizationMachine, self).__init__()

        self.user_length = user_length
        self.item_length  = item_length
        self.feature_length = feature_length

        self.hs = hs
        self.ip = ip
        self.dr = dr

        self.dropout2 = nn.Dropout(p=self.dr)  # dropout ratio
        self.qonly = qonly  # only use quadratic form
        self.command = command
        self.old_new = old_new

        # dimensions
        self.emb_size = emb_size

        print('Feature length is: {}'.format(self.feature_length))

        # _______ User embedding + Item embedding
        self.ui_emb = nn.Embedding(user_length + item_length + 1, emb_size + 1, sparse=False)

        # _______ Feature embedding and Preference embedding are common_______
        self.feature_emb = nn.Embedding(self.feature_length + 1, emb_size + 1, padding_idx=self.feature_length, sparse=False)

        # _______ Scala Bias _______
        self.Bias = nn.Parameter(torch.randn(1).normal_(0, 0.01), requires_grad=True)

        self.init_weight()


    def init_weight(self):
        self.ui_emb.weight.data.normal_(0, 0.01)
        self.feature_emb.weight.data.normal_(0, self.ip)

        # _______ set the padding to zero _______
        self.feature_emb.weight.data[self.feature_length, :] = 0

    def forward(self, ui_pair, feature_index, preference_index):
        '''
        param: a list of user ID and busi ID
        '''
        if self.command in [4, 7, 9, 10]:
            return self.forward_1234(ui_pair, feature_index, preference_index)
        else:
            return self.forward_124(ui_pair, feature_index, preference_index)


    def forward_1234(self, ui_pair, feature_index, preference_index):
        feature_matrix_feature = self.feature_emb(feature_index)
        feature_matrix_ui = self.ui_emb(ui_pair)  # (bs, 19, emb_size+1), 19 is the largest padding
        nonzero_matrix_ui = feature_matrix_ui[..., :-1]  # (bs, 19, emb_size)
        feature_bias_matrix_ui = feature_matrix_ui[..., -1:]  # (bs, 2, 1)

        nonzero_matrix_feature = feature_matrix_feature[..., :-1]
        feature_bias_matrix_feature = feature_matrix_feature[..., -1:]
        feature_matrix_preference = self.feature_emb(preference_index)
        # _______ do the dropout when passing in already _______
        nonzero_matrix_preference = feature_matrix_preference[..., :-1]  # (bs, 2, emb_size)
        feature_bias_matrix_preference = feature_matrix_preference[..., -1:]  # (bs, 2, 1)
        # _______ concatenate them together ______


        # TODO: Debug purpose
        nonzero_matrix = torch.cat((nonzero_matrix_ui, nonzero_matrix_preference), dim=1)
        feature_bias_matrix = torch.cat((feature_bias_matrix_ui, feature_bias_matrix_preference), dim=1)
        #nonzero_matrix = torch.cat((nonzero_matrix_ui, nonzero_matrix_feature, nonzero_matrix_preference), dim=1)
        #feature_bias_matrix = torch.cat((feature_bias_matrix_ui, feature_bias_matrix_feature, feature_bias_matrix_preference), dim=1)

        # _______ make a clone _______
        nonzero_matrix_clone = nonzero_matrix.clone()
        feature_bias_matrix_clone = feature_bias_matrix.clone()

        # _________ sum_square part _____________
        summed_features_embedding_squared = nonzero_matrix.sum(dim=1, keepdim=True) ** 2  # (bs, 1, emb_size)

        # _________ square_sum part _____________
        squared_sum_features_embedding = (nonzero_matrix * nonzero_matrix).sum(dim=1, keepdim=True)  # (bs, 1, emb_size)

        # ________ FM __________
        FM = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)  # (bs, 1, emb_size)

        #TODO: Still need to ADD A DROPOUT LAYER here?
        FM = self.dropout2(FM)  # (bs, 1, emb_size)
        #Bilinear = FM.sum(dim=2, keepdim=False)
        #result = Bilinear + self.Bias

        #TODO: Do it in the new way
        if self.old_new == 'new':
            new_non_zero = torch.cat((nonzero_matrix_feature, nonzero_matrix_preference), dim=1)
            summed_features_embedding_squared_new = new_non_zero.sum(dim=1, keepdim=True) ** 2
            squared_sum_features_embedding_new = (new_non_zero * new_non_zero).sum(dim=1, keepdim=True)
            newFM = 0.5 * (summed_features_embedding_squared_new - squared_sum_features_embedding_new)
            newFM = self.dropout2(newFM)
            new_non_zero_1 = nonzero_matrix_feature
            summed_features_embedding_squared_new_1 = new_non_zero_1.sum(dim=1, keepdim=True) ** 2
            squared_sum_features_embedding_new_1 = (new_non_zero_1 * new_non_zero_1).sum(dim=1, keepdim=True)
            newFM_1 = 0.5 * (summed_features_embedding_squared_new_1 - squared_sum_features_embedding_new_1)
            newFM_1 = self.dropout2(newFM_1)
            new_non_zero_2 = nonzero_matrix_preference
            summed_features_embedding_squared_new_2 = new_non_zero_2.sum(dim=1, keepdim=True) ** 2
            squared_sum_features_embedding_new_2 = (new_non_zero_2 * new_non_zero_2).sum(dim=1, keepdim=True)
            newFM_2 = 0.5 * (summed_features_embedding_squared_new_2 - squared_sum_features_embedding_new_2)
            newFM_2 = self.dropout2(newFM_2)

            Bilinear = (FM + newFM - newFM_1 - newFM_2).sum(dim=2, keepdim=False)
            result = Bilinear + self.Bias
        else:
            # TODO: Do it in the old way.
            product = torch.matmul(nonzero_matrix_feature, nonzero_matrix_preference.transpose(1, 2))
            #product = self.dropout2(product)
            product_ = product.sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
            Bilinear = FM.sum(dim=2, keepdim=False)  # (bs, 1)
            result = Bilinear + product_ + self.Bias # (bs, 1)

        return result, feature_bias_matrix_clone, torch.cat((nonzero_matrix_clone, nonzero_matrix_feature), dim=1)
    # end def

    def forward_124(self, ui_pair, feature_index, preference_index):
        feature_matrix_ui = self.ui_emb(ui_pair)  # (bs, 19, emb_size+1), 19 is the largest padding
        nonzero_matrix_ui = feature_matrix_ui[..., :-1]  # (bs, 19, emb_size)
        feature_bias_matrix_ui = feature_matrix_ui[..., -1:]  # (bs, 2, 1)

        if self.command in [1, 2, 5]:
            feature_matrix_preference = self.feature_emb(feature_index)
        elif self.command in [6, 8]:
            feature_matrix_preference = self.feature_emb(preference_index)

        # _______ dropout has been done already (when data was passed in) _______
        nonzero_matrix_preference = feature_matrix_preference[..., :-1]  # (bs, 2, emb_size)
        feature_bias_matrix_preference = feature_matrix_preference[..., -1:]  # (bs, 2, 1)

        # _______ concatenate them together ______
        nonzero_matrix = torch.cat((nonzero_matrix_ui, nonzero_matrix_preference), dim=1)
        feature_bias_matrix = torch.cat((feature_bias_matrix_ui, feature_bias_matrix_preference), dim=1)

        # _______ make a clone _______
        nonzero_matrix_clone = nonzero_matrix.clone()
        feature_bias_matrix_clone = feature_bias_matrix.clone()

        # _________ sum_square part _____________
        summed_features_embedding_squared = nonzero_matrix.sum(dim=1, keepdim=True) ** 2  # (bs, 1, emb_size)

        # _________ square_sum part _____________
        squared_sum_features_embedding = (nonzero_matrix * nonzero_matrix).sum(dim=1, keepdim=True)  # (bs, 1, emb_size)

        # ________ FM __________
        FM = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)  # (bs, 1, emb_size)

        # TODO: to remove the inter-group interaction
        # ***---***

        new_non_zero_2 = nonzero_matrix_preference
        summed_features_embedding_squared_new_2 = new_non_zero_2.sum(dim=1, keepdim=True) ** 2
        squared_sum_features_embedding_new_2 = (new_non_zero_2 * new_non_zero_2).sum(dim=1, keepdim=True)
        newFM_2 = 0.5 * (summed_features_embedding_squared_new_2 - squared_sum_features_embedding_new_2)
        FM = (FM - newFM_2)

        # ***---***

        # TODO: Still need to ADD A DROPOUT LAYER here?
        FM = self.dropout2(FM)  # (bs, 1, emb_size)

        Bilinear = FM.sum(dim=2, keepdim=False)  # (bs, 1)
        result = Bilinear + self.Bias  # (bs, 1)

        return result, feature_bias_matrix_clone, nonzero_matrix_clone
    # end def