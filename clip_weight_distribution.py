import os

import fpdf
import matplotlib.pyplot as plt
import numpy as np
import utils

BUCKETS = 16


def main():
    model, processor = utils.load_clip_model()
    model = utils.quantize_model(model)
    state = model.state_dict()

    sub_groups = {
        'mlp': ['fc1', 'fc2'],
        'self_attn': ['k_proj', 'v_proj', 'q_proj']
    }

    sensitivity = {
        ('text_model', 'mlp'): {'most': 0, 'least': 11},
        ('text_model', 'self_attn'): {'most': 0, 'least': 2},
        ('vision_model', 'mlp'): {'most': 3, 'least': 11},
        ('vision_model', 'self_attn'): {'most': 0, 'least': 10}
    }

    for encoder in ['text_model', 'vision_model']:
        for group in ['mlp', 'self_attn']:
            data = {}
            for block_index in sensitivity[(encoder, group)].values():
                output = {i: 0 for i in range(BUCKETS)}
                for subgroup in sub_groups[group]:
                    name = f'{encoder}.encoder.layers.{block_index}.{group}.{subgroup}._packed_params._packed_params'
                    add_weight_distribution(state[name], output)
                data[block_index] = output

            create_graph(encoder, group, data)


def add_weight_distribution(parameters, buckets):
    weights, _ = parameters
    weights = weights.int_repr()
    height, width = weights.shape
    for y in range(height):
        for x in range(width):
            weight = int(weights[y, x])
            bucket_index = (weight + 128) // len(buckets)
            buckets[bucket_index] += 1


def create_graph(encoder, group, data):
    encoder_name = 'Vision' if encoder == 'vision_model' else 'Text'
    group_name = 'MLP' if group == 'mlp' else 'SELF_ATTN'
    title = f'{encoder_name} Encoder {group_name} Weight Distribution'
    bucket_ranges = [(i, i + BUCKETS - 1) for i in range(-128, 127, BUCKETS)]

    plt.figure(figsize=(12, 6.5), dpi=80)
    plt.gca().ticklabel_format(axis='y', style='plain')
    columns = list(str(x) for x in range(BUCKETS))
    index = np.arange(len(columns))
    cm = plt.get_cmap('tab20b')
    for i, row in enumerate(data.keys()):
        sensitivity = "Most" if i == 0 else "Least"
        plt.plot(index, data[row].values(), color=cm(i / len(data)), marker='o',
                 label=f'Encoder Block Index {row} ({sensitivity} Sensitive)')

    plt.title(title)
    plt.xticks(range(BUCKETS), [f'{i[0]}\nto\n{i[1]}' for i in bucket_ranges])
    plt.xlabel('Weight Value Range')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'./weightdistributionplots/{title}.png')
    plt.show()


if __name__ == '__main__':
    main()
