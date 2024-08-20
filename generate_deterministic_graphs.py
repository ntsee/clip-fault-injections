import json
import os
import matplotlib.pyplot as plt
import utils


def main():
    for encoder in ['vision_model', 'text_model']:
        for group in ['mlp', 'self_attn']:
            create_graph_set(encoder, group)


def create_super_title(encoder, group):
    encoder_text = 'Vision Encoder' if encoder == 'vision_model' else 'Text Encoder'
    group_text = 'MLP' if group == 'mlp' else 'ATN'
    return f'{encoder_text} {group_text} Deterministic Faults (sorted per bucket) on CIFAR10'


def transformation_text(transformation_index):
    match transformation_index:
        case 0:
            return 'Small Negative (Flip MSB) (Sort High to Low)'
        case 1:
            return 'Small Positive (Flip 2nd MSB) (Sort Low to High)'
        case 2:
            return 'Large Positive (Flip 2nd MSB) (Sort Low to High)'
        case 3:
            return 'Large Positive (Flip MSB) (Sort Low to High)'
    return ''


def create_graph_set(encoder, group):
    fig, axs = plt.subplots(5, figsize=(10, 15), sharey=True)
    title = create_super_title(encoder, group)
    fig.suptitle(title)
    fig.supylabel('Accuracy')

    axs[-1].set_xlabel('Encoder Block Index')
    cm = plt.get_cmap('tab20b')
    for i, transformation_index in enumerate(range(4)):
        axs[i].set_title(transformation_text(transformation_index))
        for j, error_rate in enumerate([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]):
            row = []
            for block_index in range(12):
                input_file_name = f'deterministic_cifar10stats/{encoder}_{block_index}_{group}_{transformation_index}_{error_rate}.json'
                if os.path.exists(input_file_name):
                    with utils.safe_open(input_file_name, 'r') as f:
                        row.append(json.load(f))
                else:
                    row.append(None)
            axs[i].plot(range(12), row, marker='o', color=cm(j / 7), label=f'{error_rate * 100}%')
            axs[i].set_xticks(range(12))

    create_combined_graph(encoder, group, cm, axs)
    fig.legend(handles=axs[1].get_lines(), loc='lower center', ncols=7, bbox_to_anchor=(0.5, 0.912))
    fig.savefig(f'{title}.png')
    plt.show()
    plt.close()


def create_combined_graph(encoder, group, cm, axs):
    for j, error_rate in enumerate([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]):
        row = []
        for block_index in range(12):
            minimum_accuracy = None
            for i, transformation_index in enumerate(range(2)):
                input_file_name = f'deterministic_cifar10stats/{encoder}_{block_index}_{group}_{transformation_index}_{error_rate}.json'
                if os.path.exists(input_file_name):
                    with utils.safe_open(input_file_name, 'r') as f:
                        minimum_accuracy = json.load(f) if minimum_accuracy is None else min(minimum_accuracy, json.load(f))
            row.append(minimum_accuracy)
        axs[4].set_title(f'Minimum Accuracy between Small Negative and Small Positive')
        axs[4].plot(range(12), row, marker='o', color=cm(j / 7), label=f'{error_rate * 100}%')
        axs[4].set_xticks(range(12))


if __name__ == '__main__':
    main()
