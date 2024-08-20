import json
import pickle
import utils
import test_deterministic_fault_maps

def generate_fault_map(encoder, block_index, group, weights):
    output = {}
    for x, y, weight, sub_group in weights:
        name = f'{encoder}.encoder.layers.{block_index}.{group}.{sub_group}._packed_params._packed_params'
        if name not in output:
            output[name] = []
        output[name].append((x, y))
    return output


def main():
    model, processor = utils.load_clip_model()
    model = utils.quantize_model(model)

    sub_group_names = {
        'mlp': ['fc1', 'fc2'],
        'self_attn': ['k_proj', 'v_proj', 'q_proj']
    }

    transformations = [
        # Transforms weights by adding constant to weights in bucket index
        # (bucket_index, transformation_constant, sort_reversed)
        # bucket_index_0 = large_negative, bucket_index_1 = small_negative,
        # bucket_index_2 = small positive, bucket_index_3 = large positive
        (1, 128, True),
        (2, 64, False),
        (3, -64, False),
        (3, -128, False)
    ]

    state = model.state_dict()
    for encoder in ['vision_model', 'text_model']:
        for block_index in range(12):
            for group, sub_groups in sub_group_names.items():
                print(f'Processing {encoder} {block_index} {group}')
                categorized_weights = {i: [] for i in range(4)}
                for sub_group in sub_groups:
                    name = f'{encoder}.encoder.layers.{block_index}.{group}.{sub_group}._packed_params._packed_params'
                    weights, _ = state[name]
                    weights = weights.int_repr()
                    height, width = weights.shape
                    for y in range(height):
                        for x in range(width):
                            weight = int(weights[y, x])
                            bucket_index = (weight + 128) // (256 // 4)
                            indexed_weight = (x, y, weight, sub_group)
                            categorized_weights[bucket_index].append(indexed_weight)

                test_deterministic_fault_maps.test_bucket_weight_values(categorized_weights)

                total_weights = sum((len(weights) for weights in categorized_weights.values()))
                for transformation_index, (category, transform_constant, sort_reversed) in enumerate(transformations):
                    weights = categorized_weights[category]
                    weights.sort(key=lambda i: i[2], reverse=sort_reversed)
                    test_deterministic_fault_maps.test_weight_order(weights, sort_reversed)

                    category_size = len(weights)
                    for target_error_rate in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
                        target_fault_count = int(target_error_rate * total_weights)
                        actual_fault_count = min(category_size, target_fault_count)
                        output = {
                            'total_weights': total_weights,
                            'target_category': category,
                            'target_error_rate': target_error_rate,
                            'target_fault_count': target_fault_count,
                            'actual_fault_count': actual_fault_count,
                            'transformation_constant': transform_constant,
                            'faults': generate_fault_map(encoder, block_index, group,
                                                         categorized_weights[category][:actual_fault_count])
                        }

                        filename = f'deterministic_maps/{encoder}_{block_index}_{group}_{transformation_index}_{target_error_rate}.json'
                        with utils.safe_open(filename, 'w') as f:
                            json.dump(output, f)

                        print(f"Generated fault maps for category={category}, error_rate={target_error_rate}, "
                              f"target_count={target_fault_count}, actual_count={actual_fault_count}")
                        if actual_fault_count < target_fault_count:
                            break


if __name__ == "__main__":
    main()
