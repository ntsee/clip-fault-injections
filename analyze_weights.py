
import json
import os
import random
import statistics
import utils
from copy import deepcopy
from evaluate_fault_maps import target_run_configs, generate_fault_state_dict
from generate_fault_maps import (create_pickle_file_name,
                                 load_fault_map_pickle,
                                 CLIPFaultInjectionConfig,
                                 CLIPWeightFault,
                                 BitFlipStrategy, TargetNetwork, TargetEncoder,
                                 generate_layer_name_regex)
from tqdm import tqdm


OUTPUT_DIRECTORY = "weight_stats"


def create_output_filename(run_config):
    encoder, network, layer, error_rate, flip_strategy = run_config
    return f"{OUTPUT_DIRECTORY}/{encoder.name}/{layer}/{network.name}/{flip_strategy.name}/{error_rate}.json"


def main():
    model, processor = utils.load_clip_model()
    quantized_model = utils.quantize_model(model)
    original_state = deepcopy(quantized_model.state_dict())
    run_configs, fault_maps_count = target_run_configs()
    run_configs = list(run_configs)

    for run_config in tqdm(run_configs):
        encoder, network, layer, error_rate, flip_strategy = run_config
        file_name = create_output_filename(run_config)
        if os.path.exists(file_name):
            print("Skipping " + file_name)
            continue

        print("Evaluating " + file_name)

        clip_config = CLIPFaultInjectionConfig(target_encoder={encoder},
                                               target_network={network},
                                               target_layers={layer},
                                               error_rate=error_rate,
                                               bit_flip_strategy=flip_strategy)
        regex = generate_layer_name_regex(clip_config)

        means = []
        stdevs = []
        for i in tqdm(range(fault_maps_count)):
            input_file_name = create_pickle_file_name(run_config, i)
            faults = load_fault_map_pickle(input_file_name)
            state = generate_fault_state_dict(original_state, faults)

            weight_list = []
            for layer_name in state:
                match = regex.match(layer_name)
                if match:
                    weights, biases = state[layer_name]
                    height, width = weights.shape
                    for y in range(height):
                        for x in range(width):
                            weight = int(weights[y][x].int_repr())
                            weight_list.append(weight)

            mean = statistics.mean(weight_list)
            stddev = statistics.pstdev(weight_list, mean)
            means.append(mean)
            stdevs.append(stddev)

        average_mean = statistics.mean(means)
        average_stddev = statistics.pstdev(stdevs)
        with utils.safe_open(file_name) as f:
            json.dump({
                'mean': average_mean,
                'stddev': average_stddev
            }, f)



if __name__ == '__main__':
    main()
