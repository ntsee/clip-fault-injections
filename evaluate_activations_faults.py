import itertools
import json
import os
import random
import datasets
import utils
from evaluate_fault_maps import evaluate_model_accuracy
from functools import reduce
from torchvision.datasets import CIFAR10


def get_module_by_name(module, name):
    names = name.split(sep='.')
    return reduce(getattr, names, module)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def run(model, processor, dataset, text, module_name, fault_percent, offset_index, output_file_name):
    if os.path.exists(output_file_name):
        ...#return

    module = get_module_by_name(model, module_name)

    def forward_end_hook(_module, _input, _output):
        print(module_name, _output.shape)
        batch_size = _output.shape[0]
        for z in range(batch_size):
            min_activation = _output[z].min()
            max_activation = _output[z].max()
            activation_size = _output[z].numel()
            fault_length = int(activation_size * fault_percent)
            fault_offset = fault_length * offset_index
            for index in range(fault_offset, fault_offset + fault_length):
                multi_dimension_index = unravel_index(index, _output[z].size())
                _output[(z, *multi_dimension_index)] = random.uniform(min_activation, max_activation)

        return _output

    hook = module.register_forward_hook(forward_end_hook)
    accuracy = evaluate_model_accuracy(model, processor, dataset, text)
    hook.remove()

    output = {
        "module_name": module_name,
        "accuracy": accuracy,
    }

    with utils.safe_open(output_file_name) as f:
        json.dump(output, f)

    print("Finished", output_file_name, accuracy)


def main():
    model, processor = utils.load_clip_model()
    model = utils.quantize_model(model)
    cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    text = [f"a photo of a {c}" for c in cifar10.classes]
    dataset = datasets.random_sample(cifar10)

    embedding_modules = {
        'text_model': ['token_embedding', 'position_embedding'],
        'vision_model': ['patch_embedding', 'position_embedding']
    }

    fault_percent = 0.01
    for layer in reversed(['text_model', 'vision_model']):
        '''
        for embedding in embedding_modules[layer]:
            for offset in range(100):
                module_name = f'{layer}.embeddings.{embedding}'
                output_file_name = f'activations/{layer}_{embedding}_{fault_percent}_{offset}.json'
                run(model, processor, dataset, text, module_name, fault_percent, offset, output_file_name)
        '''
        for block_index in [0, 5, 8]:
            module_name = f'{layer}.encoder.layers.{block_index}.mlp.activation_fn'
            m = get_module_by_name(model, f'{layer}.encoder.layers.{block_index}.mlp')
            m_fc1 = get_module_by_name(model, f'{layer}.encoder.layers.{block_index}.mlp.fc1')
            m_fc2 = get_module_by_name(model, f'{layer}.encoder.layers.{block_index}.mlp.fc2')
            m_activation_fn = get_module_by_name(model, f'{layer}.encoder.layers.{block_index}.mlp.activation_fn')
            for offset in range(100):
                output_file_name = f'activations/{layer}_{block_index}_{fault_percent}_{offset}.json'
                run(model, processor, dataset, text, module_name, fault_percent, offset, output_file_name)

    # base accuracy is .818





if __name__ == '__main__':
    main()
