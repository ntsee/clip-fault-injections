from copy import deepcopy
from itertools import product

from PIL import Image

from generate_fault_maps import (CLIPWeightFault,
                                 TargetEncoder,
                                 TargetNetwork,
                                 BitFlipStrategy,
                                 run_config_file_name,
                                 create_pickle_file_name,
                                 load_fault_map_pickle,
                                 create_run_configs)
from datasets import ImageNet1000
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import json
import os
import random
import torch
import utils


OUTPUT_DIRECTORY = "imagenet1000stats"


def signed_flip_bit(weight: int, bit_position: int):
    mask = 1 << bit_position
    flipped_weight = (weight ^ mask) & 0xFF
    return (flipped_weight + 128) % 256 - 128


def generate_fault_state_dict(original_state, faults: list[CLIPWeightFault]):
    state = deepcopy(original_state)
    for fault in faults:
        weights, biases = state[fault.layer_name]
        weight = weights[fault.y][fault.x]
        faulted_weight = weight.dequantize() / weight.q_scale() - weight.q_zero_point()
        faulted_weight = signed_flip_bit(int(faulted_weight), fault.bit)
        faulted_weight = (faulted_weight + weight.q_zero_point()) * weight.q_scale()
        faulted_weight = torch.quantize_per_tensor(torch.tensor(faulted_weight),
                                                   scale=weight.q_scale(),
                                                   zero_point=weight.q_zero_point(),
                                                   dtype=torch.qint8)
        weights[fault.y][fault.x] = faulted_weight

    return state


def evaluate_model_accuracy(model, processor, dataset, text):
    images, target_classes = zip(*dataset)
    inputs = processor(text=text, images=list(images), return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probabilities, indices = outputs.logits_per_image.softmax(dim=1).topk(1)
    accuracy = sum(target_classes[i] == int(indices[i]) for i in range(len(indices))) / len(indices)
    return accuracy


def create_output_filename(run_config):
    encoder, network, layer, error_rate, flip_strategy = run_config
    return f"{OUTPUT_DIRECTORY}/{encoder.name}/{layer}/{network.name}/{flip_strategy.name}/{error_rate}.json"


def target_run_configs():
    encoders = {TargetEncoder.VISION}
    networks = {TargetNetwork.MLP, TargetNetwork.SELF_ATTENTION}
    layers = range(12)
    error_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    flip_strategies = [BitFlipStrategy.RANDOM]
    return product(encoders, networks, layers, error_rates, flip_strategies), 20


class ImageNet100(Dataset):

    def __init__(self, path="./imagenet100"):
        with open(f'{path}/Labels.json', 'r') as f:
            labels = json.load(f)

        self.path = path
        self.classes = [label for label in labels.values()]
        self.data = []
        for i, (key, label) in enumerate(labels.items()):
            for filename in os.listdir(f"{path}/val.X/{key}"):
                file_path = f"{path}/val.X/{key}/{filename}"
                self.data.append((file_path, i))

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        image = Image.open(file_path)
        return image, label

    def __len__(self):
        return len(self.data)


def main():
    data = ImageNet1000()
    text = [f"a photo of {c}" for c in data.classes]

    model, processor = utils.load_clip_model()
    quantized_model = utils.quantize_model(model)
    original_state = deepcopy(quantized_model.state_dict())
    run_configs, fault_maps_count = target_run_configs()
    run_configs = list(run_configs)
    for run_config in tqdm(run_configs):
        encoder, network, layer, error_rate, flip_strategy = run_config
        file_name = create_output_filename(run_config)
        if not os.path.exists(file_name):
            print("Evaluating " + file_name)
            accuracies = {}
            dataset = random.sample(list(data), 1000)
            for i in tqdm(range(fault_maps_count)):
                input_file_name = create_pickle_file_name(run_config, i)
                state = generate_fault_state_dict(original_state, load_fault_map_pickle(input_file_name))
                quantized_model.load_state_dict(state)
                quantized_model.eval()
                accuracies[i] = evaluate_model_accuracy(quantized_model, processor, dataset, text)
                gc.collect()

            with utils.safe_open(file_name) as f:
                json.dump(accuracies, f)
        else:
            print("Skipping " + file_name)


if __name__ == '__main__':
    main()

