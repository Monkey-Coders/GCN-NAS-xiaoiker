import os
import json

path = "experiment"
layers = 4
to = f"architectures_{layers}"

with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)

push = {}
        
for model_hash, model in architectures.items():
    if len(model["weights"]) == layers:
        push[model_hash] = {
            "weights": model["weights"],
            "val_acc": model["val_acc"],
            "time": model["time"],
        } 
        
with open(f"{to}/generated_architectures.json", "w") as f:
    json.dump(push, f)
