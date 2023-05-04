import os
import json



base_path = "experiment"


with open("experiment/correlations/correlations.json", "r") as f:
    correlation = json.load(f)
    proxies = list(correlation[list(correlation.keys())[0]].keys())
    proxies.sort()
    for proxy in proxies:
        text = f"\cellcolor{'{lightgray}'} {proxy}"
        for epoch in correlation.keys():
            text += f" & {correlation[epoch][proxy]:.4f}"
            
        print(text + " \\\\ \hline")