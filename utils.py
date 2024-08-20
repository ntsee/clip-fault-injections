import os
import torch
from transformers import CLIPProcessor, CLIPModel


def print_size_of_model(model):
    torch.save(model.state_dict(), 'temp.p')
    print(f"Size (MB): {os.path.getsize('temp.p')/1e6}")


def load_clip_model():
    device = "cpu"
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', ).eval().to(torch.device(device))
    torch.set_grad_enabled(False)
    return model, processor


def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def safe_open(path, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)
