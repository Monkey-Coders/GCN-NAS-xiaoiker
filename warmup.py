import argparse
from collections import OrderedDict
import pickle
import shutil
import os
from ZeroCostFramework.zero_cost_controller import calculate_zc_proxy_scores
import yaml
import torch
import inspect
import numpy as np
import random
import torch.nn as nn
import scipy
import json

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(path):
    config = get_config(path)
    Feeder = import_class(config["feeder"])
    data_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**config["train_feeder_args"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_worker"],
        drop_last=True,
        worker_init_fn=init_seed)
    return data_loader
        

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_config(path):
    config_file = f"{path}/work_dir/config.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def initialize_model(path, file_name):
    # Get the config file from the model path
    config = get_config(path)
    model_args = config["model_args"]
    #print(model_args)
    output_device = config["device"][0]
    Model = import_class(config["model"])
    shutil.copy2(inspect.getfile(Model), config["work_dir"])
    #print(Model)
    model = Model(**model_args).cuda(output_device)
    #print(model)
    
    # Find all files within the folder that ends with .pt
    if file_name is None:
        return model
    weights = torch.load(f"{path}/{file_name}")
        
    weights = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights.items()])
    try:
        model.load_state_dict(weights)
        return model
    except:
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print('Can not find these weights:')
        for d in diff:
            print('  ' + d)
        state.update(weights)
        model.load_state_dict(state)
        return None

    
def get_zc_scores(path, file_name, overide = []):
    model = initialize_model(path, file_name)
    if model is None:
        raise Exception("Model is None")
    data_loader = load_data(path)
    config = get_config(path)
    device = config["device"][0]
    loss_function = nn.CrossEntropyLoss().cuda(device)
    overide = overide

    scores = calculate_zc_proxy_scores(model, data_loader, device, loss_function, overide)
    return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--hash", type=str, default="", required=True)
    parser.add_argument("--path", type=str, default="experiment", required=False)
    parser.add_argument("--overide", type=str, default="", required=False)
    args = parser.parse_args()
    model_hash = str(args.hash)
    base_path = str(args.path)
    overide_arg = str(args.overide)
    if overide_arg == "":
        overide_arg = []
    else:
        overide_arg = overide_arg.split("|")
    print("Model hash: ", model_hash)
    try:
        pt_files = [f for f in os.listdir(f"{base_path}/run/{model_hash}") if f.endswith(".pt")]
    except:
        with open(f"{base_path}/run_not_found.txt", "a") as f:
            f.write(model_hash + "\n")
                    
        exit()
        
    pt_files.sort()
    track_scores = {}
    
    with open(f"{base_path}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
        
    # if "zero_cost_scores" not in architectures[model_hash]:
    print(f"Calculating zero cost scores for epoch {-1}...")
        
    scores = get_zc_scores(f"{base_path}/run/{model_hash}", None, overide_arg)
    track_scores[f"zero_cost_scores"] = scores
    
    for file in pt_files:
        epoch = int(file.split("-")[1]) 
        if epoch > 10:
            continue
        if f"zero_cost_scores_{epoch}" in architectures[model_hash]:
            continue
        print(f"Calculating zero cost scores for epoch {epoch}...")
        try:
            scores = get_zc_scores(f"{base_path}/run/{model_hash}", file, overide_arg)
        except Exception as e:
            print(f"Error: file not found")
            continue
        track_scores[f"zero_cost_scores_{epoch}"] = scores


    with open(f"{base_path}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
    temp_archi_dict = architectures[model_hash]
    for key in track_scores:
        try:
            temp_archi_dict[key] = {**architectures[model_hash][key], **track_scores[key]}
        except:
            temp_archi_dict[key] = track_scores[key]
    architectures[model_hash] = temp_archi_dict
    with open(f"{base_path}/generated_architectures.json", "w") as f:
        json.dump(architectures, f)