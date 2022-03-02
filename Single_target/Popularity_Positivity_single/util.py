import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import pprint
from matplotlib import cm
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties


class Dataset:
    def __init__(self, lastpath="../../data/FM-train-data"):
        self.lastpath = lastpath

    def load_file(self, filename, change_lastpath=False, lastpath=""):
        if not change_lastpath:
            lastpath = self.lastpath

        with open(os.path.join(lastpath, filename), 'r') as f:
            data = json.load(f)
        return data

    def save_file(self, data, filename, change_lastpath=False, lastpath=""):
        if not change_lastpath:
            lastpath = self.lastpath
        with open(os.path.join(lastpath, filename), 'w') as f:
            json.dump(data, f)


class Visualization:
    @staticmethod
    def count_user_positive(user_positive, range=50, lastpath="figures", filename="numbers_of_user_liked_items"):
        count = [len(l) for x, l in user_positive.items()]

        fig = plt.figure(figsize=(4.5, 3))
        ax1 = fig.add_axes([0.15, 0.2, 0.7, 0.7])
        sns.distplot(count, bins=np.arange(0, max(count), range), kde=False, hist_kws=dict(ec="k"), ax=ax1)
        gca = plt.gca()
        gca.set_title("Number of users liked items.")
        gca.set_xlabel("numbers of liked items")
        gca.set_ylabel("Number of users")

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                         ha='center', va='bottom', rotation=45, fontsize=5)  # set the alignment of the text

        filepath = os.path.join(lastpath, filename)
        # plt.savefig(filepath + ".eps", format='eps')
        plt.savefig(filepath + ".pdf", format='pdf')

    @staticmethod
    def count_item_feature(item_feature, lastpath="figures", filename1="feature_statistics",
                           filename2="number_attributes_per_item"):
        attribute_sequence = list(itertools.chain(*[list(x.values())[0] for x in item_feature.values()]))
        fig = plt.figure(figsize=(6.5, 3))
        ax1 = fig.add_axes([0.15, 0.2, 0.8, 0.7])
        sns.countplot(attribute_sequence, color="C0", ax=ax1)
        plt.xticks(rotation=50, fontsize=9)

        gca = plt.gca()
        gca.set_title("Statistics of attributes.")
        gca.set_xlabel("Item ID")
        gca.set_ylabel("Count")

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                         ha='center', va='bottom', rotation=45, fontsize=5)  # set the alignment of the text

        filepath = os.path.join(lastpath, filename1)
        # plt.savefig(filepath + ".eps", format='eps')
        plt.savefig(filepath + ".pdf", format='pdf')

        # %%
        number_attribute_per_item = [len(list(x.values())[0]) for x in item_feature.values()]
        fig = plt.figure(figsize=(6.5, 3))
        ax1 = fig.add_axes([0.15, 0.2, 0.8, 0.7])
        sns.countplot(number_attribute_per_item, ax=ax1)
        plt.xticks(rotation=50, fontsize=9)

        gca = plt.gca()
        gca.set_title("Number of attributes per item.")
        gca.set_xlabel("Number of attributes")
        gca.set_ylabel("Count of items")

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                         ha='center', va='bottom', rotation=45, fontsize=5)  # set the alignment of the text

        filepath = os.path.join(lastpath, filename2)
        # plt.savefig(filepath + ".eps", format='eps')
        plt.savefig(filepath + ".pdf", format='pdf')

    @staticmethod
    def illustrate_item_distribution(popularity, feature, filename, lastpath="figures"):
        populariy = list(popularity.values())

        fig = plt.figure(figsize=(6.5, 3))
        ax1 = fig.add_axes([0.15, 0.2, 0.7, 0.7])

        sns.distplot(populariy, bins=np.arange(0, max(populariy), max(populariy) / 20), kde=False,
                     hist_kws=dict(ec="k"), ax=ax1)

        gca = plt.gca()
        gca.set_title("{} of items".format(feature.capitalize()))
        gca.set_xlabel("{}".format(feature.capitalize()))
        gca.set_ylabel("Count of items")

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                         ha='center', va='bottom', rotation=45, fontsize=4)  # set the alignment of the text

        filepath = os.path.join(lastpath, filename)
        # plt.savefig(filepath + ".eps", format='eps')
        plt.savefig(filepath + ".pdf", format='pdf')


class Math:
    @staticmethod
    def zipf(x, s=0.5, alpha=1):
        nominator = alpha * x ** -s
        denominator = 1
        return nominator / denominator

    @staticmethod
    def pareto(x, xm=1, alpha=1):
        return (alpha * xm ** alpha) / (x ** (alpha + 1))


def reset_popularity_based_on_defined_function(item_exposure_distribution):
    item_popularity = {k: v for k, v in
                       sorted(item_exposure_distribution.items(), key=lambda item: item[1], reverse=True)}
    rank = range(len(item_popularity))
    redefined_popularity = [Math.zipf(x + 1) for x in rank]
    redefined_popularity = np.array(redefined_popularity)
    redefined_popularity = redefined_popularity / sum(redefined_popularity)
    redefined_item_popularity = dict(zip([int(i) for i in item_popularity.keys()], redefined_popularity))
    return redefined_item_popularity


def get_redefined_popularity_from_original_exposure_distribution(data):

    if not hasattr(data,'item_exposure_distribution'):
        data.item_exposure_distribution = data.load_file('item_exposure_distribution.json')

    # Transform the bad popularity to a predefined popularity, reserving their orders.
    data.redefined_item_popularity = reset_popularity_based_on_defined_function(data.item_exposure_distribution)

    # Fixing the missing value in item exposure file:
    print("============================================\nFixing the missing value in item exposure file:")
    item_id_small_rawID = data.load_file('busi_list_test.json')
    # Dictionary from raw id to id.

    photo_trans = data.load_file("photo_trans.json")
    # small data's bigID in [0, 10728]
    data.item_id_small_bigID = [photo_trans[str(photo)] for photo in item_id_small_rawID]

    print("The number of items in small data is [{}]".format(len(item_id_small_rawID)))
    print("However, the number of items in exposure distribution is [{}]".format(len(data.item_exposure_distribution)))
    mising_item_set = set(data.item_id_small_bigID) - set([int(i) for i in data.item_exposure_distribution.keys()])
    print("The missing item id is [{}]".format(mising_item_set))
    print("Set the missing items' popularity to the popularity of the random existing items.")
    for i in mising_item_set:
        random_key = random.sample(data.redefined_item_popularity.keys(), 1)[0]
        data.redefined_item_popularity[i] = data.redefined_item_popularity[random_key]
        print("Set the popularity of item id:[{}] to the popularity of item id:[{}]".format(i, random_key))

    print("Normalized the distribution again.")
    values = np.array(list(data.redefined_item_popularity.values()))
    values_norm = values / sum(values)
    data.redefined_item_popularity = dict(zip(data.redefined_item_popularity.keys(), values_norm))

    redefined_popularity_name = 'redefined_item_popularity.json'
    print("Saved the file to '{}'".format(os.path.join(data.lastpath, redefined_popularity_name)))
    data.save_file(data.redefined_item_popularity, redefined_popularity_name)

if __name__ == '__main__':

    data = Dataset(lastpath="../../data/FM-train-data")

    data.user_positive = data.load_file('review_dict_test.json')
    data.item_feature = data.load_file('item_dict-merge.json')
    data.item_exposure_distribution = data.load_file('item_exposure_distribution.json')
    data.item_positive_distribution = data.load_file('item_positive_distribution.json')

    Visualization.count_user_positive(data.user_positive, 50, filename="numbers_of_user_liked_items")
    Visualization.count_item_feature(data.item_feature, filename1="feature_statistics",
                                     filename2="number_attributes_per_item")

    # This is for the selection bias:
    Visualization.illustrate_item_distribution(data.item_positive_distribution,
                                               "User selection (positive sample)",
                                               filename='item_positive_distribution')

    # This is for the popular bias:
    Visualization.illustrate_item_distribution(data.item_exposure_distribution,
                                               "Popularity (exposure)",
                                               filename='item_exposure_distribution')

    get_redefined_popularity_from_original_exposure_distribution(data)

    # Illustrate the redefined popularity distribution.
    Visualization.illustrate_item_distribution(data.redefined_item_popularity,
                                               "Popularity (exposure) - (redefined)",
                                               filename="redefined_popularity_item_distribution")

    plt.show()
