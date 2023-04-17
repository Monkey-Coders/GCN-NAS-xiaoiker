import os
import json

path = "architectures_10"
# to = "architectures_10"

with open(f"{path}/generated_architectures.json", "r") as f:
    architectures = json.load(f)


print(architectures.keys())
        
