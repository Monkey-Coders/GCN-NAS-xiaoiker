import argparse
from subprocess import call
import json
import time
import os
import yaml
path = "architectures/generated_architectures.json"

default_configs = {
    
    # feeder
    "feeder": "feeders.feeder.Feeder",
    "train_feeder_args": {
        "data_path": "../max_data_out/ntu/xview/train_data_joint.npy",
        "label_path": "../max_data_out/ntu/xview/train_label.pkl",
        "debug": False,
        "random_choose": True,
        "random_shift": False,
        "random_move": False,
        "window_size": 270,
        "normalization": False,
    },
    "test_feeder_args": {
        "data_path": "../max_data_out/ntu/xview/val_data_joint.npy",
        "label_path": "../max_data_out/ntu/xview/val_label.pkl",
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
    "num_epoch": 30,
    "nesterov": True,
}

# Implement parser
def get_parser():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--start", type=str, default=0, required=True)
    parser.add_argument("--end", type=str, default=25, required=True)
    return parser



if __name__ == "__main__":
    # Get the architectures from the path
    print("Running train_random_architectures.py")
    parser = get_parser()
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)
    with open(path, "r") as f:
        architectures = json.load(f)
    # Loop through the dictionary of architectures
    for i, (model_hash, model) in enumerate(architectures.items()):
        if i < start:
            continue
        if i > end:
            break
        # Check if model contains "val_acc"
        if "val_acc" not in model:
            # Create a config file
            config = default_configs
            config["model_args"]["weights"] = model["weights"]
            config["model"] = "model.dynamic_model.Model"
            config["work_dir"] = f"architectures/run/{model_hash}/work_dir"
            config["model_saved_name"] = f"architectures/run/{model_hash}/runs"
            # Save the config file as a yaml file in the work_dir
            # Create folder architectures/configs if it does not exist
            if not os.path.exists("architectures/configs"):
                os.makedirs("architectures/configs")
            
            with open(f"architectures/configs/{model_hash}.yaml", "w") as f:
                yaml.dump(config, f)
            # Sleep for 1 second
            time.sleep(2)
            #call(["python3", "train.py", f"architectures/configs/{model_hash}.yaml"])
            command = f"python3 main.py --config architectures/configs/{model_hash}.yaml"
            print("Calling command: ", command)
            call(command, shell=True)
