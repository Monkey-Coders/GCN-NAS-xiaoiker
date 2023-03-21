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

path = "architectures/run/2ecc95faf6febc1db5ca5b2a06eed301a46289493dbe6fae5a94d5228e806432"
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
        print("*"*50)
        print(set(weights.keys()))
        print("*"*50)
        print('Can not find these weights:')
        for d in diff:
            print('  ' + d)
        state.update(weights)
        model.load_state_dict(state)

    
    


if __name__ == "__main__":
    val_acc = 0.7876054852320675
    print(f"Val acc: {val_acc}")
    model = initialize_model(path, "runs-1-1896.pt")
    data_loader = load_data(path)
    config = get_config(path)
    device = config["device"][0]
    loss_function = nn.CrossEntropyLoss().cuda(device)
    save_path = "test"

    scores = calculate_zc_proxy_scores(model, data_loader, device, loss_function, save_path)
    print(scores)
    

    model = initialize_model(path, "runs-3-3792.pt")
    data_loader = load_data(path)
    config = get_config(path)
    device = config["device"][0]
    loss_function = nn.CrossEntropyLoss().cuda(device)
    save_path = "test"

    scores = calculate_zc_proxy_scores(model, data_loader, device, loss_function, save_path)
    print(scores)

    model = initialize_model(path, "runs-5-5688.pt")
    data_loader = load_data(path)
    config = get_config(path)
    device = config["device"][0]
    loss_function = nn.CrossEntropyLoss().cuda(device)
    save_path = "test"

    scores = calculate_zc_proxy_scores(model, data_loader, device, loss_function, save_path)
    print(scores)

    model = initialize_model(path, "runs-7-7584.pt")
    data_loader = load_data(path)
    config = get_config(path)
    device = config["device"][0]
    loss_function = nn.CrossEntropyLoss().cuda(device)
    save_path = "test"

    scores = calculate_zc_proxy_scores(model, data_loader, device, loss_function, save_path)
    print(scores)



