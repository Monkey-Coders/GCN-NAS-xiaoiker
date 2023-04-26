import os

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
path = "output"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and ".out" in f]
x = []
for file in files:
    with open(f"{path}/{file}", "r") as f:
        lines = f.read()
        if "PermissionError" in lines:
            x.append(file)
print(x)
        