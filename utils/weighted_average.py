import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import pandas as pd
import scipy

def load_data(path):
    with open(f'{path}/generated_architectures.json') as f:
        return json.load(f)

def create_legend(weight_colors):
    handles = []
    for size, c in weight_colors.items():
        handles.append(mpatches.Patch(color=c, label=f'{size}'))
    return handles

def main_func(path, data, weight_colors, epoch, legend_handles):
    scores = list(data.values())[0][epoch].keys()
    zc_scores = {score: [] for score in scores}
    normalized = {}
    val_acc = []
    colors = []

    for (model_hash, model) in data.items():
        # if len(model["weights"]) != 10:
        #     continue
        if epoch not in model:
            continue
        val_acc.append(model["val_acc"])
        colors.append(weight_colors[len(model["weights"])])
        for score in scores:
            zc_scores[score].append(float(model[epoch][score]["score"]) if model[epoch][score]["score"] != "Nan" else float("nan"))

    for (method, zc_score) in zc_scores.items():
        normalized[method] = [(float(i)-min(zc_score))/(max(zc_score)-min(zc_score)) for i in zc_score]

    corr_matrix = calculate_correlation(val_acc, normalized, scores)
    weighted_average = wa_func(val_acc, normalized, scores, corr_matrix)
    plotis(val_acc, weighted_average, colors, legend_handles, path)
    
    
def calculate_correlation(val_acc, normalized, scores):

    _data = []
  

    for i in range(len(val_acc)):
      _d = {"val_acc": val_acc[i]}
      for score in scores:
        _d[score] = normalized[score][i]
      _data.append(_d)

    df = pd.DataFrame(_data)
    corr_matrix = df.corr()
    dict_corr_matrix = dict(corr_matrix["val_acc"])
    return dict_corr_matrix
  
def wa_func(val_acc, normalized, scores, corr_matrix):
    """# Weighted Average"""
    override_scores = []
    if len(override_scores) > 0:
      scores = override_scores
      
    weighted_average = []
    for i in range(len(val_acc)):
      wa = 0
      for score in scores:
          wa += normalized[score][i] * corr_matrix[score]
      weighted_average.append(wa)
    return weighted_average
  
def plotis(val_acc, weighted_average, colors, hand, path):
    for i in range(len(val_acc)):
      plt.scatter(val_acc[i], weighted_average[i], c=colors[i])
    plt.xlabel('val_acc')
    plt.ylabel('weighted_average')
    spr_rank = scipy.stats.spearmanr(val_acc, weighted_average)
    print(spr_rank)
    plt.title(f'Weighted Average\nspearmanrank={spr_rank[0]:.3f}')
    plt.legend(handles=hand, loc='upper left')
    plt.savefig(f"{path}/plot/weighted_average.png")


if __name__ == '__main__':
    path = "experiment"
    data = load_data(path)
    
    weight_colors = {
        4: "red",
        6: "teal",
        8: "olive",
        10: "magenta"
    }
    legend_handles = [mpatches.Patch(color=c, label=f'{size}') for size, c in weight_colors.items()]

    os.makedirs(f"{path}/plot", exist_ok=True)
    epoch = "zero_cost_scores"
    main_func(path, data, weight_colors, epoch, legend_handles)





