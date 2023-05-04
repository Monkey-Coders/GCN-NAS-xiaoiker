import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy

def load_data(path):
    with open(f'{path}/generated_architectures.json') as f:
        architectures = json.load(f)
        temp_arch = {}
        for i, (model_hash, model) in enumerate(architectures.items()):
            path_to_weights = f"{path}/run/{model_hash}"
            files = os.listdir(path_to_weights)
            weights = [file for file in files if file.endswith(".pt")]
            if len(weights) == 0:
                continue
            max_epoch = 0
            for weight in weights:
                epoch = int(weight.split("-")[1])
                if epoch > max_epoch:
                    max_epoch = epoch
            if max_epoch >= 45 and model["val_acc"] > 0.85:
                temp_arch[model_hash] = model
        
        return temp_arch

def create_legend(weight_colors):
    handles = []
    for size, c in weight_colors.items():
        handles.append(mpatches.Patch(color=c, label=f"{size}"))
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
            zc_scores[score].append(
                float(model[epoch][score]["score"])
                if model[epoch][score]["score"] != "Nan"
                else float("nan")
            )

    for (method, zc_score) in zc_scores.items():
        normalized[method] = [
            (float(i) - min(zc_score)) / (max(zc_score) - min(zc_score))
            for i in zc_score
        ]

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

    corr_matrix = {}
    for score in scores:
        corr_matrix[score] = scipy.stats.spearmanr([obj[score] for obj in _data], val_acc)[0]
    return corr_matrix

def wa_func(val_acc, normalized, scores, corr_matrix):
    """# Weighted Average"""
    override_scores = []
    if len(override_scores) > 0:
        scores = override_scores
    weighted_average = []
    for i in range(len(val_acc)):
        wa = 0
        weighted_sum = 0
        for score in scores:
            wa += normalized[score][i] * abs(corr_matrix[score])
            weighted_sum += abs(corr_matrix[score])
        wa /= weighted_sum
        weighted_average.append(wa)
    return weighted_average


def plotis(val_acc, weighted_average, colors, hand, path):
    plt.scatter(val_acc, weighted_average)
    m, b = np.polyfit(val_acc, weighted_average, 1) 
    line_y = m*np.array(val_acc) + b
    plt.plot(val_acc, line_y,'r:')

    plt.xlabel("val_acc")
    plt.ylabel("weighted_arithmetic_mean")
    spr_rank = scipy.stats.spearmanr(val_acc, weighted_average)
    plt.title(f"Weighted Arithmetic Mean\nspearmanrank={spr_rank[0]:.3f}")
    # plt.legend(handles=hand, loc="upper left")
    plt.savefig(f"{path}/plot/weighted_average.png")


if __name__ == "__main__":
    path = "experiment"
    data = load_data(path)

    weight_colors = {4: "red", 6: "teal", 8: "olive", 10: "magenta"}
    legend_handles = [
        mpatches.Patch(color=c, label=f"{size}") for size, c in weight_colors.items()
    ]

    os.makedirs(f"{path}/plot", exist_ok=True)
    epoch = "zero_cost_scores"
    main_func(path, data, weight_colors, epoch, legend_handles)
