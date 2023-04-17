import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

def load_data(path):
    with open(f'{path}/generated_architectures.json') as f:
        return json.load(f)

def normalize(lst):
    mx = max(lst)
    mn = min(lst)
    return (mx - mn)

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
        if epoch not in model:
            continue
        val_acc.append(model["val_acc"])
        colors.append(weight_colors[len(model["weights"])])
        for score in scores:
            zc_scores[score].append(float(model[epoch][score]["score"]) if model[epoch][score]["score"] != "Nan" else float("nan"))

    for (method, zc_score) in zc_scores.items():
        nmz = normalize(zc_score)
        normalized[method] = [zc_s / nmz for zc_s in zc_score]

    ncols = 4
    nrows = math.ceil(len(scores) / ncols)
    handles = create_legend(weight_colors)

    plot_all_in_one("all_in_one", path, scores, val_acc, normalized, score_colors)
    
    proxies = ["flops", "jacov", "zen"]
    plot_vote("vote", path, val_acc, normalized, proxies, colors, handles)
    plot_proxies("zc_proxies", path, scores, nrows, ncols, val_acc, zc_scores, colors, handles)
    plot_proxies("normalized", path, scores, nrows, ncols, val_acc, normalized, colors, handles)

def plot_proxies(filename, path, scores, nrows, ncols, val_acc, scores_data, colors, handles):
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for i, score in enumerate(scores):
        ax = axs[i // ncols, i % ncols]
        ax.set_title(score)
        for a in range(len(val_acc)):
            ax.scatter(val_acc[a], scores_data[score][a], color=colors[a])
        ax.set_xlabel('val_acc')
        ax.set_ylabel(score)

    fig.legend(handles=handles, loc='upper right')
    fig.tight_layout(pad=2.0)
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")

def plot_all_in_one(filename, path, scores, val_acc, normalized, score_colors):
    plt.clf()
    plt.cla()
    plots = {score: plt.scatter(val_acc, normalized[score], marker='o', color=score_colors[score]) for i, score in enumerate(scores)}

    plt.legend(plots.values(), plots.keys(), loc='upper right')
    plt.xlabel('val_acc')
    plt.ylabel('normalized score')
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")

def plot_vote(filename, path, val_acc, normalized, proxies, colors, handles):
    plt.clf()
    plt.cla()
    vote = [normalized[p] for p in proxies]
    y = np.average(vote, axis=0)

    for i in range(len(val_acc)):
        plt.scatter(val_acc[i], y[i], color=colors[i])
    plt.xlabel('val_acc')
    plt.ylabel('vote')
    plt.title(f"{proxies}")
    plt.rcParams["figure.figsize"] = (8, 8)

    plt.legend(handles=handles, loc='upper right')
    plt.savefig(f"{path}/plot/{epoch}/{filename}.png")


if __name__ == '__main__':
    path = "experiment"
    data = load_data(path)
    score_colors = {
        "plain": "blue",
        "params": "red",
        "flops": "green",
        "synflow": "orange",
        "snip": "purple",
        "grad_norm": "brown",
        "epe_nas": "pink",
        "grasp": "gray",
        "fisher": "cyan",
        "l2_norm": "black",
        "jacov": "magenta",
        "zen": "olive",
        "nwot": "teal",
        "grad_sign": "navy"
    }
    score_markers = {
        "plain": "o",
        "params": "x",
        "flops": "-",
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
    for i in range(10):
        epochs.append(f"zero_cost_scores_{i}")
    
    for epoch in epochs:
        os.makedirs(f"{path}/plot/{epoch}", exist_ok=True)
        plot_scores(path, data, weight_colors, score_colors, score_markers, epoch)


