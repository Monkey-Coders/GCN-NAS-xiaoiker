import os
import json

path = "experiment"


with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)
    times = []
    for i, (model_hash, model) in enumerate(architectures.items()):
        times.append(model["time"])

print(f"Average time: {sum(times)/len(times)}")
print(f"Total time: {sum(times)}")


