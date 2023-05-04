import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

def load_data(path):
    with open(f'{path}/generated_architectures.json') as f:
        return json.load(f)
        # temp_arch = {}
        # for i, (model_hash, model) in enumerate(architectures.items()):
        #     path_to_weights = f"{path}/run/{model_hash}"
        #     files = os.listdir(path_to_weights)
        #     weights = [file for file in files if file.endswith(".pt")]
        #     if len(weights) == 0:
        #         continue
        #     max_epoch = 0
        #     for weight in weights:
        #         epoch = int(weight.split("-")[1])
        #         if epoch > max_epoch:
        #             max_epoch = epoch
        #     if max_epoch >= 45:
        #         temp_arch[model_hash] = model
        
        # return temp_arch

def create_legend(weight_colors):
    handles = []
    for size, c in weight_colors.items():
        handles.append(mpatches.Patch(color=c, label=f'{size}'))
    return handles

def plot_scores(path, data, weight_colors, score_colors, score_markers, epoch):
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

    ncols = 4
    nrows = math.ceil(len(scores) / ncols)
    handles = create_legend(weight_colors)

    plot_all_in_one("all_in_one", path, scores, val_acc, normalized, score_colors, score_markers)
    
    proxies = ['synflow', 'zen', 'plain']
    plot_vote("vote", path, val_acc, normalized, proxies, colors, handles, score_markers, score_colors)
    plot_proxies("zc_proxies", path, scores, nrows, ncols, val_acc, zc_scores, colors, handles, score_markers)
    plot_proxies("normalized", path, scores, nrows, ncols, val_acc, normalized, colors, handles, score_markers)

def plot_proxies(filename, path, scores, nrows, ncols, val_acc, scores_data, colors, handles, score_markers):
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for i, score in enumerate(scores):
        ax = axs[i // ncols, i % ncols]
        ax.set_title(score)
        ax.scatter(val_acc, scores_data[score])
        ax.set_xlabel('val_acc')
        ax.set_ylabel(score)

    # fig.legend(handles=handles, loc='upper right')
    fig.tight_layout(pad=2.0)
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")

def plot_all_in_one(filename, path, scores, val_acc, normalized, score_colors, score_markers):
    plt.clf()
    plt.cla()
    plots = {score: plt.scatter(val_acc, normalized[score], color=score_colors[score]) for i, score in enumerate(scores)}

    plt.legend(plots.values(), plots.keys(), loc='upper right')
    plt.xlabel('val_acc')
    plt.ylabel('normalized score')
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")

def plot_vote(filename, path, val_acc, normalized, proxies, colors, handles, score_markers, score_colors):
    plt.clf()
    plt.cla()
    handle = []
    
    for p in proxies:
        handle.append(mpatches.Patch(color=score_colors[p], label=f'{p}'))
        plt.scatter(val_acc, normalized[p], color=score_colors[p])
        
    plt.xlabel('val_acc')
    plt.ylabel('vote')
    plt.title(",".join(proxies))
    plt.rcParams["figure.figsize"] = (8, 8)

    plt.legend(handles=handle, loc='upper right')
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")


if __name__ == '__main__':
    path = "experiment"
    data = load_data(path)
    
    score_colors = {
        "params": "red",
        "flops": "orange",
        "grad_norm": "brown",
        "epe_nas": "blue",
        "grasp": "gray",
        "l2_norm": "black",
        "jacov": "magenta",
        "fisher": "olive",
        "nwot": "cyan",
        "grad_sign": "navy",
        "snip": "purple",
        "plain": "orange",
        "synflow": "green",
        "zen": "blue",
    }
    score_markers = {
        "plain": "o",
        "params": "x",
        "flops": "_",
        "synflow": "s",
        "snip": "p",
        "grad_norm": "P",
        "epe_nas": "+",
        "grasp": "<",
        "fisher": ">",
        "l2_norm": "^",
        "jacov": "v",
        "zen": ".",
        "nwot": "D",
        "grad_sign": "X"
    }
    weight_colors = {
        4: "red",
        6: "teal",
        8: "olive",
        10: "magenta"
    }
    epochs = ["zero_cost_scores"]
    # for i in range(10):
    #     epochs.append(f"zero_cost_scores_{i}")
    
    for epoch in epochs:
        os.makedirs(f"{path}/plot/{epoch}", exist_ok=True)
        plot_scores(path, data, weight_colors, score_colors, score_markers, epoch)


