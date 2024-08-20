import os
import random
import utils
from evaluate_fault_maps import evaluate_model_accuracy
from torchvision.datasets import CIFAR10
from evaluate_fault_maps import ImageNet1000
from tqdm import tqdm


def main():
    cifar10 = ImageNet1000()# CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    text = [f"a photo of {c}" for c in cifar10.classes]

    model, processor = utils.load_clip_model()
    quantized_model = utils.quantize_model(model)
    accuracy = 0
    for _ in tqdm(range(20)):
        dataset = random.sample(list(cifar10), 1000)
        accuracy += evaluate_model_accuracy(quantized_model, processor, dataset, text)

    print("Average Accuracy", accuracy / 20)


if __name__ == "__main__":
    main()
