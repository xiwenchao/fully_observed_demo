
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Dataset:
    def __init__(self):
        with open('../../data/FM-train-data/test_photo_record.json', 'r') as f:
            self.popularity = json.load(f)

class Math:
    @staticmethod
    def zipf(x, s=0.5, alpha=1):
        nominator = alpha * x ** -s
        denominator = 1
        return nominator / denominator

    @staticmethod
    def pareto(x, xm=1, alpha=1):
        return (alpha * xm ** alpha) / (x ** (alpha + 1))



class Visualization:
    @staticmethod
    def count_popular(popularity, lastpath="figures", filename="popularity_items"):
        populariy = list(popularity.values())

        fig = plt.figure(figsize=(4.5, 3))
        ax1 = fig.add_axes([0.15, 0.2, 0.7, 0.7])
        sns.displot(populariy, bins=np.arange(0, max(populariy), max(populariy) / 20), kde=False)
        gca = plt.gca()
        gca.set_title("Populariy of items")
        gca.set_xlabel("Popularity")
        gca.set_ylabel("Count of items")

        for p in ax1.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                         ha='center', va='bottom', rotation=45, fontsize=4)  # set the alignment of the text

        filepath = os.path.join(lastpath, filename)
        # plt.savefig(filepath + ".eps", format='eps')
        plt.savefig(filepath + ".pdf", format='pdf')

def resort_popularity_based_on_defined_function(data):
    item_popularity = {k: v for k, v in sorted(data.popularity.items(), key=lambda item: item[1], reverse=True)}
    rank = range(len(item_popularity))
    redefined_popularity = [Math.zipf(x+1) for x in rank]
    redefined_popularity = np.array(redefined_popularity)
    redefined_popularity = redefined_popularity / sum(redefined_popularity)
    redefined_item_popularity = dict(zip(item_popularity.keys(), redefined_popularity))
    data.redefined_item_popularity = redefined_item_popularity



if __name__ == '__main__':
    data = Dataset()
    Visualization.count_popular(data.popularity)
    resort_popularity_based_on_defined_function(data)
    Visualization.count_popular(data.redefined_item_popularity, filename="redefined_popularity_items")
    plt.show()

global_data = Dataset()
resort_popularity_based_on_defined_function(global_data)