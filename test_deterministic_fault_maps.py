def test_bucket_weight_values(categorized_weights: dict[int, list[tuple[int, int, int, int]]]):
    expected_ranges = {
        0: (-128, -65),
        1: (-64, -1),
        2: (0, 63),
        3: (64, 127)
    }

    for category, weights in categorized_weights.items():
        min_val, max_val = expected_ranges[category]
        for x, y, weight, sub_group in weights:
            assert min_val <= weight <= max_val


def test_weight_order(weights: list[tuple[int, int, int, int]], reverse: bool):
    def less_or_equal(a, b):
        return a <= b

    def greater_or_equal(a, b):
        return a >= b

    comparator = less_or_equal if not reverse else greater_or_equal

    last_value = None
    for x, y, weight, sub_group in weights:
        if last_value:
            assert comparator(last_value, weight)
        last_value = weight
