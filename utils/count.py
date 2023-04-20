import json

D = {
    "total": 0,
}

path = "experiment"

with open(f"{path}/generated_architectures.json") as f:
    architectures = json.load(f)
    
for architecture in architectures.values():
    L = len(architecture["weights"])
    if str(L) not in D:
        D[str(L)] = 0
    D[str(L)] += 1
    D["total"] += 1
    
print(D)