import os
import json

path = "experiment"
layers = 4
to = f"architectures_{layers}"

def has_nan(d):
    for k, v in d.items():
        if isinstance(v, dict):
            if has_nan(v):
                return True
        elif isinstance(v, float) and float(v) == float('nan'):
            return True
        elif v == "nan":
            return True
    return False

with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)

push = {}
        
for model_hash, model in architectures.items():
    if has_nan(model):
        continue
    push[model_hash] = model 
        
with open(f"{path}/generated_architectures.json", "w") as f:
    json.dump(push, f)
