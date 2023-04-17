import numpy as np
from utils import import_class
import json
import hashlib
import argparse
import os

path = "architectures_10"

def get_model_hash(model):
    model_hash = hashlib.sha256(repr(model).encode()).hexdigest()
    return model_hash

def generate_random_weights(layers, operations):
    matrix = np.random.uniform(0.05, 0.15, size=(layers, operations))
    for i in range(layers):
        last_three = matrix[i, -3:]
        if not any(last_three > 0.1):
            last_three[np.random.randint(0, 3)] = np.random.uniform(0.1, 0.15)
        matrix[i, -3:] = last_three
    return matrix


def generate_model(weights):

    model_args = {
        'num_class': 60, 
        'num_point': 25,
        'num_person': 2,
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {
            'labeling_mode': 'spatial'
        },
        "weights": weights,
    }
    output_device = 0

    model_label = "model.dynamic_model.Model"
    Model = import_class(model_label)
    model = Model(**model_args).cuda(output_device)
    return model


def store_model(model, weights):
    model_hash = get_model_hash(model)
    try:
        with open(f"{path}/generated_architectures.json", "r") as f:
            architectures = json.load(f)
    except FileNotFoundError:
        architectures = {}
    # If model is not in the file add it
    if model_hash not in architectures:
        print("Model does not exist, adding it to the file")
        architectures[model_hash] = {
            "weights": weights.tolist(),
        }
        with open(f"{path}/generated_architectures.json", "w") as f:
            json.dump(architectures, f)
    else:
        print("Model already exists")


parser = argparse.ArgumentParser()
parser.add_argument("--architectures", type=int, default=80)
parser.add_argument("--layers", type=int, default=10)
parser.add_argument("--operations", type=int, default=8)

# Main function
if __name__ == "__main__":
    args = parser.parse_args()
    architectures = args.architectures
    layers = args.layers
    operations = args.operations
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({}, f)
    print(f"Starting to generate {architectures} random architectures")
    for i in (range(architectures)):
        weights = generate_random_weights(layers, operations)
        model = generate_model(weights)
        store_model(model, weights)