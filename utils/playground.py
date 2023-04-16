import os
import json

path = "experiment"

with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)

a = {}

for model_hash, model in architectures.items():
    for method, score_time in model["zero_cost_scores"].items():
        if score_time["score"] == "NaN" or score_time["time"] == "NaN" or score_time["score"] == 0:
            if model_hash not in a:
                a[model_hash] = []
            a[model_hash].append(method)
    for i in range(10):
        try:
            for method, score_time in model[f"zero_cost_scores_{i}"].items():
                if score_time["score"] == "NaN" or score_time["time"] == "NaN" or score_time["score"] == 0:
                    if model_hash not in a:
                        a[model_hash] = []
                    a[model_hash].append(method)
                    
        except KeyError:
            print(f"KeyError: {model_hash} {i}")
    
    if model_hash in a:
        a[model_hash] = list(set(a[model_hash]))
        
print(a)