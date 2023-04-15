import os
import json

path = "experiment"

with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)

for model_hash, model in architectures.copy().items():
    if len(model["weights"]) != 10:
        del architectures[model_hash]

with open(f"{path}/generated_architectures.json", "w") as f:
    json.dump(architectures, f)