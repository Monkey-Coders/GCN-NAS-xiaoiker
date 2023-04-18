# get all the data from experiments/generated_architectures.json and format them so that we have a table with zero cost scores for each epoch for each proxy for each architecture in a pandas table
# Path: utils/report_data.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ZeroCostFramework.utils.util_functions import get_proxies

base_path = "experiment"

epochs = ["zero_cost_scores"]

for i in range(9):
    epochs.append(f"zero_cost_scores_{i+1}")

def validate_architecture(architecture, epoch):
    return all(format in architecture for format in ["val_acc", epoch])

if __name__ == '__main__':

    zero_cost_proxies = get_proxies()
    with open(f'{base_path}/generated_architectures.json') as f:
        architectures = json.load(f)

    data = []
    for architecture in architectures:
        for epoch in epochs:
            if validate_architecture(architectures[architecture], epoch):
                for proxy in zero_cost_proxies:
                    if proxy in architectures[architecture][epoch]:
                        data.append([architecture, epoch, proxy, architectures[architecture][epoch][proxy]["score"], architectures[architecture]["val_acc"]])
    df = pd.DataFrame(data, columns=["architecture", "epoch", "proxy", "score", "val_acc"])
    df.to_csv(f"{base_path}/data.csv", index=False)
    # Plot
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.lineplot(x="epoch", y="score", hue="proxy", data=df)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Epoch")
    plt.ylabel("Zero Cost Score")
    plt.tight_layout()
    plt.savefig(f"{base_path}/report/zero_cost_scores.png")
    # Plot
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.lineplot(x="epoch", y="val_acc", data=df)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(f"{base_path}/report/val_acc.png")

