import random

from PIL import Image
from torch.utils.data import Dataset
import os


class ImageNet1000(Dataset):

    def __init__(self, path=f"./imagenet1000"):
        self.path = path
        with open(f'{path}/words.txt', 'r') as f:
            labels = {}
            for line in f:
                i, name = line.strip('\n').split('\t')
                labels[i] = name

        self.classes = []
        self.data = []
        class_indexes = {}
        for folder_name in os.listdir(f'{path}/val'):
            if folder_name not in class_indexes:
                class_indexes[folder_name] = len(class_indexes)
                self.classes.append(labels[folder_name])

            i = class_indexes[folder_name]
            for filename in os.listdir(f'{path}/val/{folder_name}'):
                file_path = f"{path}/val/{folder_name}/{filename}"
                self.data.append((file_path, i))

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        image = Image.open(file_path)
        return image, label

    def __len__(self):
        return len(self.data)


def random_sample(dataset, num_samples=1000, seed=0):
    state = random.getstate()
    random.seed(seed)
    dataset = random.sample(list(dataset), num_samples)
    random.setstate(state)
    return dataset