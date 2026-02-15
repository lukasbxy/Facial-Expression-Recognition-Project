#!/usr/bin/env python3
"""
Script for performing  activation maximization to visualize what a given channel in a model is looking for.

Sample usage
python -m activation_maximization \
  --ckpt path/to/checkpoint.pt \
  --module layer3.0 \
  --channels 0, 2, 4, 8, 16, 32 \
  --init-image path/to/image.jpg \
  --topk 150 \
  --outdir activation_maximization_out
"""

import os
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

# use non-interactive matplotlib backend (for servers / ssh)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as T

from models import ResNet18_SE_Variant, ResNet18, ResNet18_SE, ResNet34, CCT


# -----------------------------
# helpers
# -----------------------------
def total_variation(x):
    """
    Total variation loss.

    Penalizes rapid changes in pixels and encourage smoother images with less high freq noise.
    """
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def normalize(x, mean, std):
    """
    Normalize image tensor using mean and standard dev (std)
    """
    mean = torch.tensor(mean, device=x.device).view(1,3,1,1)
    std = torch.tensor(std, device=x.device).view(1,3,1,1)
    return (x - mean) / std


def tensor_to_img(x):
    """
    Convert tensor image (1,3,H,W) to numpy image (H,W,3)
    for saving with matplotlib.
    """
    x = x.detach().clamp(0,1)[0]
    return x.permute(1,2,0).cpu().numpy()


def save_image(img_np, path, title):
    """
    Save numpy image using matplotlib.
    """
    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_init_image(path, size, device):
    """
    Load an image from disk and convert to tensor.
    may be used as initialization instead of random noise.
    """
    tfm = T.Compose([
        T.Resize((size,size)),
        T.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


# -----------------------------
# acces module
# -----------------------------
def get_module(model, name):
    """
    Get a module from the model using a string name.

    Example:
        "layer3.0.conv1"
    """
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


class Hook:
    """
    Forward hook to capture activations.
    """
    def __init__(self, module):
        self.act = None
        self.h = module.register_forward_hook(self.fn)

    def fn(self, module, inp, out):
        self.act = out

    def close(self):
        self.h.remove()


# -----------------------------
# infer channels
# -----------------------------
@torch.no_grad()
def infer_channels(model, module_name, image_size, device):
    """
    Run dummy input through model to figure ut number of channels
    in the target module.
    """
    module = get_module(model, module_name)
    hook = Hook(module)

    # dummy input img
    x = torch.zeros(1,3,image_size,image_size,device=device)

    _ = model(x)

    act = hook.act
    hook.close()

    return act.shape[1]


# -----------------------------
# objective
# -----------------------------
def topk_objective(act_ch, topk):
    """
    Calculate objective using top-k absolute activations.

    This focuses on strongest responding regions.
    """

    flat = act_ch.flatten(1)

    k = min(topk, flat.shape[1])

    vals = torch.topk(flat.abs(), k=k, dim=1).values

    return vals.mean()


# -----------------------------
# optimization
# -----------------------------
def maximize_channel(
    model,
    module_name,
    channel_idx,
    image_size,
    device,
    init_img,
    topk,
    # internal hyperparameters
    # expose these if you want to experiment with them
    steps=400,
    lr=0.02,
    l2_weight=1e-4,
    tv_weight=5e-3,
    blur_every=25,
    jitter_px=4,
):

    module = get_module(model, module_name)
    hook = Hook(module)

    if init_img is None:
        param = torch.randn(1,3,image_size,image_size,
                            device=device, requires_grad=True)
    else:
        # initialize from real image (convert to logit space)
        eps = 1e-6
        img0 = init_img.clamp(eps,1-eps)
        param = torch.log(img0/(1-img0)).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([param], lr=lr)

     # normalization constants (ImageNet)
    mean = (0.485,0.456,0.406)
    std  = (0.229,0.224,0.225)

    for step in range(steps):

        opt.zero_grad()

        # convert param to image using sigmoid
        img = torch.sigmoid(param)

        # random spatial jitter
        if jitter_px > 0:
            dx = random.randint(-jitter_px, jitter_px)
            dy = random.randint(-jitter_px, jitter_px)
            img = torch.roll(img, shifts=(dy,dx), dims=(2,3))

        # normalize image before feeding to model
        x = normalize(img, mean, std)

        _ = model(x)

        act = hook.act

        # select desired channel
        act_ch = act[:, channel_idx]

        # compute objective
        obj = topk_objective(act_ch, topk)

        # regularization losses
        l2 = img.pow(2).mean()
        tv = total_variation(img)

        # total loss (negative because we maximize)
        loss = -obj + l2_weight*l2 + tv_weight*tv

        loss.backward()
        opt.step()

        # blur occasionally
        if blur_every > 0 and (step+1) % blur_every == 0:
            with torch.no_grad():
                img2 = torch.sigmoid(param)
                img2 = F.avg_pool2d(img2,3,1,1)
                eps = 1e-6
                img2 = img2.clamp(eps,1-eps)
                param.data = torch.log(img2/(1-img2))

    hook.close()

    return torch.sigmoid(param).detach()


# -----------------------------
# main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    # currently only important args are exposed, but feel free to add more if you want to experiment with different things
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--module", type=str, default="layer3.0")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--init_img", type=str, default="")
    parser.add_argument("--channels", type=str, default="0,1,2")
    parser.add_argument("--topk", type=int, default=100)

    args = parser.parse_args()

    # choose gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.outdir, exist_ok=True)

    model = ResNet18_SE_Variant()

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state"])
        print("loaded checkpoint")

    model = model.to(device).eval()

    # optional init image
    init_img = None
    if args.init_img:
        init_img = load_init_image(args.init_img,
                                   args.image_size,
                                   device)

    # channel list
    if args.channels == "all":

        C = infer_channels(
            model,
            args.module,
            args.image_size,
            device
        )

        channel_list = list(range(C))

    else:
        channel_list = [int(x) for x in args.channels.split(",")]

    for ch in channel_list:

        print("maximizing channel", ch)

        img = maximize_channel(
            model,
            args.module,
            ch,
            args.image_size,
            device,
            init_img,
            args.topk
        )

        img_np = tensor_to_img(img)

        path = os.path.join(
            args.outdir,
            f"{args.module.replace('.','_')}_ch{ch}.png"
        )

        save_image(img_np, path,
                   f"{args.module} ch{ch}")

    print("done")


if __name__ == "__main__":
    main()
