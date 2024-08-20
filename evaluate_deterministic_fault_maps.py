import json
import os
import pickle
import random

import torch
from copy import deepcopy
from datasets import ImageNet1000
from evaluate_fault_maps import evaluate_model_accuracy
from torchvision.datasets import CIFAR10
import utils


def main():
    model, processor = utils.load_clip_model()
    model = utils.quantize_model(model)

    random.seed(0)
    k = 1000
    cifar10 = ImageNet1000() # CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    text = [f"a photo of a {c}" for c in cifar10.classes]
    dataset = random.sample(list(cifar10), k)
    images, target_classes = zip(*dataset)

    original_state = model.state_dict()
    for encoder in ['vision_model', 'text_model']:
        for block_index in range(12):
            for group in ['mlp', 'self_attn']:
                for error_rate in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
                    for transformation_index in range(4):
                        input_filename = f'deterministic_maps/{encoder}_{block_index}_{group}_{transformation_index}_{error_rate}.json'
                        if not os.path.exists(input_filename):
                            continue

                        output_filename = f'deterministic_imagenet1000stats/{encoder}_{block_index}_{group}_{transformation_index}_{error_rate}.json'
                        if os.path.exists(output_filename):
                            continue

                        print(f"Evaluating {input_filename}...")

                        with utils.safe_open(input_filename, 'r') as f:
                            fault_map = json.load(f)

                        transformation_constant = fault_map['transformation_constant']
                        state = deepcopy(original_state)
                        for state_name, faults in fault_map['faults'].items():
                            weights, _ = state[state_name]
                            for x, y in faults:
                                weight = weights[y, x]
                                faulted_weight = int(weight.int_repr()) + transformation_constant
                                faulted_weight = (faulted_weight + weight.q_zero_point()) * weight.q_scale()
                                weights[y, x] = torch.quantize_per_tensor(torch.tensor(faulted_weight),
                                                                          scale=weight.q_scale(),
                                                                          zero_point=weight.q_zero_point(),
                                                                          dtype=torch.qint8)

                        model.load_state_dict(state)
                        model.eval()

                        accuracy = evaluate_model_accuracy(model, processor, dataset, text)
                        with utils.safe_open(output_filename, 'w') as f:
                            json.dump(accuracy, f)


if __name__ == '__main__':
    main()
