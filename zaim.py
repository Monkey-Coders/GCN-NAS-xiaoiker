import argparse
from subprocess import call
import json
import time
import os
import yaml

default_configs = {
    # feeder
    "feeder": "feeders.feeder.Feeder",
    "train_feeder_args": {
        "data_path": "./data/ntu/xview/train_data_joint.npy",
        "label_path": "./data/ntu/xview/train_label.pkl",
        "debug": False,
        "random_choose": False,
        "random_shift": False,
        "random_move": False,
        "window_size": -1,
        "normalization": False,
    },
    "test_feeder_args": {
        "data_path": "./data/ntu/xview/val_data_joint.npy",
        "label_path": "./data/ntu/xview/val_label.pkl",
    },
    "model": None,
    "model_args": {
        "num_class": 60,
        "num_point": 25,
        "num_person": 2,
        "graph": "graph.ntu_rgb_d.Graph",
        "graph_args": {
            "labeling_mode": "spatial",
        },
        "weights": None,
    },
    #optim
    "weight_decay": 0.0006,
    "base_lr": 0.1,
    "step": [30, 45, 60],
    # training
    "device": [0],
    "batch_size": 40,
    "test_batch_size": 20,
    "num_epoch": 10,
    "nesterov": True,
}

# Implement parser
def get_parser():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--hash", type=str, default="", required=False)
    parser.add_argument("--path", type=str, default="experiment", required=False)
    parser.add_argument("--weights", type=str, default="", required=False)
    parser.add_argument("--start_epoch", type=str, default="", required=False)
    return parser



if __name__ == "__main__":
    # Get the architectures from the path
    print("Running train_random_architectures.py")
    parser = get_parser()
    args = parser.parse_args()
    model_hash = str(args.hash)
    path = str(args.path)
    weights = str(args.weights)
    start_epoch = str(args.start_epoch)
    with open(f"{path}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
    # Loop through the dictionary of architectures
    
    model = architectures[model_hash]


    # for i, (model_hash, model) in enumerate(architectures.items()):
    
        # Check if there exists a folder with the model_hash as name in the {path}/run folder

    # Create a config file
    config = default_configs
    if weights != "":
        config["model_args"]["weights"] = weights
    else:
        config["model_args"]["weights"] = model["weights"]
        
    config["model"] = "model.dynamic_model.Model"
    config["work_dir"] = f"{path}/run/{model_hash}/work_dir"
    config["model_saved_name"] = f"{path}/run/{model_hash}/runs"
    config["model_hash"] = model_hash
    config["save_path"] = path
    # Save the config file as a yaml file in the work_dir
    # Create folder architectures/configs if it does not exist
    if not os.path.exists(f"{path}/configs"):
        os.makedirs(f"{path}/configs")
    
    with open(f"{path}/configs/{model_hash}.yaml", "w") as f:
        yaml.dump(config, f)
    # Sleep for 1 second
    time.sleep(2)
    command = f"python3 main_2.py --config {path}/configs/{model_hash}.yaml"
    if start_epoch != "":
        command += f" --start-epoch {start_epoch}"
    print("Calling command: ", command)
    call(command, shell=True)
