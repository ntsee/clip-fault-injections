import json
import matplotlib.pyplot as plt
import numpy as np
import utils
from generate_fault_maps import (TargetEncoder, TargetNetwork, BitFlipStrategy)
from evaluate_fault_maps import create_output_filename

OUTPUT_DIRECTORY = "imagenetplots"


def main():
    baseline_accuracy = round( 0.31404999999999994 * 100, 2)
    encoders = [TargetEncoder.VISION]
    networks = [TargetNetwork.MLP, TargetNetwork.SELF_ATTENTION]
    flip_strategies = [BitFlipStrategy.RANDOM]
    encoder_indexes = range(12)
    error_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for encoder in encoders:
        for network in networks:
            for flip_strategy in flip_strategies:
                data = [[baseline_accuracy for _ in encoder_indexes]]
                for error_rate in error_rates:
                    row = []
                    for encoder_index in encoder_indexes:
                        run_config = encoder, network, encoder_index, error_rate, flip_strategy
                        print(create_output_filename(run_config))
                        with utils.safe_open(create_output_filename(run_config), "r") as f:
                            accuracies = json.load(f)
                            accuracy = sum(accuracies.values()) / len(accuracies)
                            row.append(round(accuracy * 100, 2))
                    data.append(row)
                generate_graph((encoder, network, flip_strategy), data)


def create_title(config):
    encoder, network, flip_strategy = config
    encoder = "[ImageNet1000 Text Encoder]" if encoder is TargetEncoder.TEXT else "[ImageNet1000] Vision Encoder"
    network = "MLP" if network is TargetNetwork.MLP else "ATN"
    flip_strategy = "Random" if flip_strategy is BitFlipStrategy.RANDOM else "Worst Case"
    return f"{encoder} {network} {flip_strategy} Faults"


def generate_graph(config, data):
    plt.figure(figsize=(6, 7.5))
    columns = list(str(x) for x in range(12))
    rows = ["no-fault", "0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%"]
    colors = ['slategrey', 'lightsteelblue', 'royalblue', 'blue', "slateblue", "darkslateblue", "indigo", "blueviolet"]
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    cell_text = []
    for row in range(n_rows):
        plt.plot(index, data[row], color=colors[row], marker='o')
        y_offset = data[row]
        cell_text.append([x for x in y_offset])

    table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylabel("Accuracy")
    title = create_title(config)
    plt.title(title)
    plt.xticks([])
    plt.yticks([i for i in range(0, 110, 10)])
    plt.xlabel('Encoder Block Index', labelpad=97)
    plt.savefig(f"{OUTPUT_DIRECTORY}/{title}.png")
    plt.close()


if __name__ == "__main__":
    main()
