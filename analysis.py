import sys
sys.path.append("..")

import argparse
from PIL import Image
from src.utils.transforms import ValTransform
from src.models.dml_module import DMLModule
from PIL import Image
from tqdm import tqdm
import torch
from trained_models import models

from src.datamodules.cub200 import Cub200
from src.datamodules.cars196 import Cars196
from src.datamodules.stanford_online_products import StanfordOnlineProducts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="", help="Model to use. Can be '' for all models or a specific model name")
parser.add_argument("--dataset", type=str, default="cars196", choices=["cars196", "cub200", "sop"], help="Dataset name")
args = parser.parse_args()

# Filter models
models = {k: v for k, v in models.items() if (args.dataset in k) and (args.model in k)}

print(f"Loading {len(models)} model checkpoints")
models = {
    name: DMLModule.load_from_checkpoint(path, map_location=device).eval() for name, path in models.items()
}

# ## Data
# since the transforms are specified for images and masks, we need to provide a dummy mask
val_transform = ValTransform()

print("Loading dataset")
datasets = {
    "cars196": Cars196,
    "cub200": Cub200,
    "sop": StanfordOnlineProducts,
}

dataset = datasets[args.dataset]("data", mode="test", transform=val_transform)

print("Creating base image embeddings")
# Base image: Creates a black image which is transformed/normalized
base_img, _ = val_transform(Image.new("RGB", (500, 500)), Image.new("L", (500, 500)))
base_embeddings = {
    name: model(base_img.unsqueeze(0).to(device)).detach() for name, model in models.items()
}

# hyperparams
num_samples = 25
std_dev_spread = 0.15
percentile = 0.99

def attention_score(attention, mask):
    image_percentage_meaningful = mask.sum() / mask.numel()
    attention_percentage_meaningful = (attention * mask).sum() / attention.sum()
    
    return (attention_percentage_meaningful - image_percentage_meaningful) / (1.0 - image_percentage_meaningful)


scores = {
    name: [] for name in models.keys()
}

print("Computing attention scores")
for image_idx in tqdm(range(len(dataset))):
    item = dataset[image_idx]
    image = item["inputs"].unsqueeze(0).to(device)
    mask = item["masks"].to(device)

    for i, (name, model) in enumerate(models.items()):
        data = image.repeat(num_samples, 1, 1, 1)
        data += torch.randn_like(data) * std_dev_spread * (data.max() - data.min())
        _ = data.requires_grad_()

        out = model(data)
        loss = model.loss_func.base_loss_func.distance(out, base_embeddings[name]).mean()
        loss.backward()

        grads = data.grad.mean(dim=1).detach()

        # process grads
        grads = grads.abs().sum(dim=0).detach()

        v_max = torch.quantile(torch.flatten(grads), percentile)
        v_min = torch.min(grads)
        attention = torch.clamp((grads - v_min) / (v_max - v_min), 0.0, 1.0)

        att_score = attention_score(attention, mask)
        scores[name].append(att_score)

print("Writing attention scores to file")
for name in models.keys():
    torch.save(scores[name], f"./attention_scores/{name.replace(' ', '_')}_attention_scores.pt")

print("Done")
