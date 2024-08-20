import json
import matplotlib.pyplot as plt
import numpy as np
import utils
from generate_fault_maps import (TargetEncoder, TargetNetwork, BitFlipStrategy)
from analyze_weights import create_output_filename
from generate_fault_maps import (create_pickle_file_name,
                                 load_fault_map_pickle,
                                 CLIPFaultInjectionConfig,
                                 CLIPWeightFault,
                                 BitFlipStrategy, TargetNetwork, TargetEncoder,
                                 generate_layer_name_regex)
import statistics
from tqdm import tqdm
OUTPUT_DIRECTORY = "plots_weights"


def create_title(config):
    encoder, network, flip_strategy = config
    encoder = "Text Encoder" if encoder is TargetEncoder.TEXT else "Vision Encoder"
    network = "MLP" if network is TargetNetwork.MLP else "ATN"
    flip_strategy = "Random" if flip_strategy is BitFlipStrategy.RANDOM else "Worst Case"
    return f"{encoder} {network} {flip_strategy}"


def create_graph(config, data, stat_name):
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
    plt.ylabel(stat_name)
    title = create_title(config)
    plt.title(title)
    plt.xticks([])
    plt.xlabel('Encoder Block Index', labelpad=97)
    plt.savefig(f"{OUTPUT_DIRECTORY}/{create_title(config)} {stat_name}.png")
    plt.close()


def main():
    model, processor = utils.load_clip_model()
    quantized_model = utils.quantize_model(model)
    state = quantized_model.state_dict()
    encoders = [TargetEncoder.TEXT, TargetEncoder.VISION]
    networks = [TargetNetwork.MLP, TargetNetwork.SELF_ATTENTION]
    flip_strategies = [BitFlipStrategy.RANDOM, BitFlipStrategy.WORST_CASE]
    encoder_indexes = range(12)
    error_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for encoder in encoders:
        for network in networks:
            for flip_strategy in flip_strategies:
                # Calculate baseline weight / mean stdd

                weight_list = []
                clip_config = CLIPFaultInjectionConfig(target_encoder={encoder},
                                                       target_network={network},
                                                       target_layers={0},
                                                       error_rate=0,
                                                       bit_flip_strategy=flip_strategy)
                regex = generate_layer_name_regex(clip_config)
                for layer_name in state:
                    match = regex.match(layer_name)
                    if match:
                        weights, biases = state[layer_name]
                        height, width = weights.shape
                        for y in range(height):
                            for x in range(width):
                                weight = int(weights[y][x].int_repr())
                                weight_list.append(weight)

                mean = round(statistics.mean(weight_list) * 1000, 3)
                stddev = round(statistics.pstdev(weight_list, mean) * 1000, 3)
                mean_data = [[mean for _ in range(12)]]
                std_data = [[stddev for _ in range(12)]]
                for error_rate in error_rates:
                    mean_row = []
                    std_row = []
                    for encoder_index in encoder_indexes:
                        run_config = encoder, network, encoder_index, error_rate, flip_strategy
                        print(create_output_filename(run_config))
                        with utils.safe_open(create_output_filename(run_config), "r") as f:
                            stats = json.load(f)
                            mean_row.append(round(stats["mean"] * 1000, 3))
                            std_row.append(round(stats["stddev"] * 1000, 3))

                    mean_data.append(mean_row)
                    std_data.append(std_row)

                create_graph((encoder, network, flip_strategy),  mean_data, "Weight Mean")
                create_graph((encoder, network, flip_strategy),  mean_data, "Weight STDDev")


if __name__ == '__main__':
    main()
