import json

with open("experiment/generated_architectures.json", "r") as f:
    architectures = json.load(f)

with open("experiment/generated_architectures_old.json", "r") as f:
    architectures_old = json.load(f)
    
count = 0
for i, (model_hash, model) in enumerate(architectures.items()):
    if architectures_old[model_hash]["val_acc"] != model["val_acc"]:
        print(f"# 'layers' : {len(model['weights'])}     - index: {i}")
        print(f"\033[92m+ 'val_acc': {model['val_acc']}\033[0m")
        print(f"\033[91m- 'val_acc': {architectures_old[model_hash]['val_acc']}\033[0m")
        count += 1

print()
print(f"Total: {count}")

