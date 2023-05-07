import os
import json


base_path = "experiment"


with open("experiment/correlations/correlations.json", "r") as f:
    correlation = json.load(f)
    proxies = list(correlation[list(correlation.keys())[0]].keys())
    proxies.sort()
    for i, proxy in enumerate(proxies):
        text = "\multicolumn{1}{l|}{"
        if i % 2 == 0:
            text += "\cellcolor{verylightgray}"
            text += f"{proxy}"
        else:
            text += f"{proxy}"
            
        text += "}"
        for epoch in correlation.keys():
            if i % 2 == 0:  
                text += f" & \cellcolor{'{verylightgray}'}${correlation[epoch][proxy]:.4f}$"
            else:
                text += f" & ${correlation[epoch][proxy]:.4f}$"
                
            break
        print(text + " \\\\")
