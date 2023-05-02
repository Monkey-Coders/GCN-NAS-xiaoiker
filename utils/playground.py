import os
import json

# path = "experiment"
# layers = 4
# to = f"architectures_{layers}"

# def has_nan(d):
#     for k, v in d.items():
#         if isinstance(v, dict):
#             if has_nan(v):
#                 return True
#         elif isinstance(v, float) and float(v) == float('nan'):
#             return True
#         elif v == "nan":
#             return True
#     return False

# with open(f"{path}/generated_architectures.json", "r") as f:
#     architectures = json.load(f)

# push = {}
        
# for model_hash, model in architectures.items():
#     if has_nan(model):
#         continue
#     push[model_hash] = model 
        
# with open(f"{path}/generated_architectures.json", "w") as f:
#     json.dump(push, f)

# get all files with .out in output folder
""" path = "output"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and ".out" in f]
x = []
for file in files:
    with open(f"{path}/{file}", "r") as f:
        lines = f.read()
        if "PermissionError" in lines:
            x.append(file)
print(x) """

base_path = "experiment"
with open(f'{base_path}/generated_architectures_test.json') as f:
        architectures = json.load(f)
        # temp_arch = {}
        # for i, (model_hash, model) in enumerate(architectures.items()):
        #     path_to_weights = f"{base_path}/run/{model_hash}"
        #     files = os.listdir(path_to_weights)
        #     weights = [file for file in files if file.endswith(".pt")]
        #     if len(weights) == 0:
        #         continue
        #     max_epoch = 0
        #     for weight in weights:
        #         epoch = int(weight.split("-")[1])
        #         if epoch > max_epoch:
        #             max_epoch = epoch
        #     if max_epoch >= 45 and model["val_acc"] > 0.85:
        #         temp_arch[model_hash] = model
        # architectures = temp_arch

# with open(f'{base_path}/generated_architectures_test.json', 'w') as f:
#     json.dump(architectures, f)

# Loop through all elements in the architectures dict and print out the index and the val_acc
# time = []
# for i, (model_hash, model) in enumerate(architectures.items()):
#     time.append(model["time"])
    
# print(f"average time: {sum(time)/len(time)}")
# print(f"total time: {sum(time)}")