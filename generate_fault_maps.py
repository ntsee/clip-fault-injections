from dataclasses import dataclass, asdict
from enum import Enum
from functools import partial
from itertools import product
from random import randint
from tqdm import tqdm
from typing import Optional
import json
import pickle
import random
import re
import utils


OUTPUT_DIRECTORY = "maps"


class BitFlipStrategy(Enum):
    RANDOM = partial(lambda: randint(0, 8))
    WORST_CASE = partial(lambda: 7)
    BEST_CASE = partial(lambda: 1)


class TargetEncoder(Enum):
    TEXT = "text_model"
    VISION = "vision_model"


class TargetNetwork(Enum):
    MLP = "mlp"
    SELF_ATTENTION = "self_attn"


@dataclass
class CLIPFaultInjectionConfig:
    target_encoder: set[TargetEncoder]
    target_network: set[TargetNetwork]
    target_layers: set[int]
    error_rate: float
    bit_flip_strategy: BitFlipStrategy = BitFlipStrategy.RANDOM
    random_seed: Optional[int] = None


@dataclass(unsafe_hash=True)
class CLIPWeightFault:
    layer_name: str
    x: int
    y: int
    bit: int


def generate_layer_name_regex(config: CLIPFaultInjectionConfig):
    encoder_name = f"(?:{'|'.join(str(x.value) for x in config.target_encoder)})" if len(
        config.target_encoder) > 1 else str(tuple(config.target_encoder)[0].value)
    layer_index = f"(?:{'|'.join(str(x) for x in config.target_layers)})" if len(config.target_layers) > 1 else str(
        *config.target_layers)
    network_name = f"(?:{'|'.join(str(x.value) for x in config.target_network)})" if len(
        config.target_network) > 1 else str(tuple(config.target_network)[0].value)
    return re.compile(
        r"^({}\.encoder\.layers\.{}\.{}\.[a-zA-Z0-9_]+\.)(:?_packed_params\._packed_params)$"
        .format(encoder_name, layer_index, network_name))


def create_fault_map(quantized_model, config: CLIPFaultInjectionConfig) -> tuple[list[CLIPWeightFault], int]:
    if config.random_seed:
        random.seed(config.random_seed)

    faults = []
    state = quantized_model.state_dict()
    regex = generate_layer_name_regex(config)
    total_weights = 0
    for name in state.keys():
        match = regex.match(name)
        if match:
            name = match.group(1) + '_packed_params._packed_params'
            weights, biases = state[name]
            height, width = weights.shape
            total_weights += width * height
            for y in range(height):
                for x in range(width):
                    if random.random() < config.error_rate:
                        bit = config.bit_flip_strategy.value()
                        faults.append(CLIPWeightFault(layer_name=name, x=x, y=y, bit=bit))

    return faults, total_weights


def save_fault_map_json(faults: list[CLIPWeightFault], file_name: str):
    with utils.safe_open(file_name, 'w') as f:
        json.dump([asdict(x) for x in faults], f)


def save_fault_map_pickle(faults: list[CLIPWeightFault], file_name: str):
    with utils.safe_open(file_name, 'wb') as f:
        pickle.dump(faults, f)


def load_fault_map_json(file_name: str) -> list[CLIPWeightFault]:
    with open(file_name, 'r') as f:
        file_contents = f.read()
        return [CLIPWeightFault(**data) for data in json.loads(file_contents)]


def load_fault_map_pickle(file_name: str) -> list[CLIPWeightFault]:
    with utils.safe_open(file_name, 'rb') as f:
        return pickle.load(f)


def create_run_configs():
    encoders = [TargetEncoder.TEXT, TargetEncoder.VISION]
    networks = [TargetNetwork.MLP, TargetNetwork.SELF_ATTENTION]
    layers = list(range(12))
    error_rates = [0.005, 0.002]
    flip_strategies = [BitFlipStrategy.RANDOM, BitFlipStrategy.WORST_CASE]
    fault_maps_count = 20
    run_configs = list(product(encoders, networks, layers, error_rates, flip_strategies))
    return run_configs, fault_maps_count


def run_config_file_name(run_config, i):
    encoder, network, layer, error_rate, flip_strategy = run_config
    return f"{encoder.name}/{layer}/{network.name}/{flip_strategy.name}/{error_rate}/{i}"


def create_pickle_file_name(run_config, i):
    return f"{OUTPUT_DIRECTORY}/{run_config_file_name(run_config, i)}.pickle"


def main():
    model, processor = utils.load_clip_model()
    quantized_model = utils.quantize_model(model)
    run_configs, fault_maps_count = create_run_configs()
    for run_config in tqdm(run_configs):
        encoder, network, layer, error_rate, flip_strategy = run_config
        config = CLIPFaultInjectionConfig(
            target_encoder={encoder},
            target_network={network},
            target_layers={layer},
            error_rate=error_rate,
            bit_flip_strategy=flip_strategy)
        for i in range(fault_maps_count):
            file_name = create_pickle_file_name(run_config, i)
            faults, total_weights = create_fault_map(quantized_model, config)
            save_fault_map_pickle(faults, file_name)


if __name__ == '__main__':
    main()
