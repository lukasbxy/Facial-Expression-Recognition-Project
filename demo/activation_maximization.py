#!/usr/bin/env python3
r"""
Script for performing  activation maximization to visualize what a given channel in a model is looking for.

How to use: 
python demo/activation_maximization.py \
  --model "resnet18_variant" \
  --ckpt "PATH/TO/CHECKPOINT" \
  --module "layer3.0" \
  --channels "0,2,4,8,16,32" \
  --init_img "PATH/TO/INITIAL/IMAGE" \
  --topk 150 \
  --outdir "PATH/TO/OUT/DIRECTORY"

"""

import os
import random
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

# use non-interactive matplotlib backend (for servers / ssh)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as T

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18_SE_Variant, ResNet18, ResNet18_SE, ResNet34, ResNet18_Variant


MODEL_REGISTRY = {
    "resnet18": ResNet18,
    "resnet18_se": ResNet18_SE,
    "resnet18_variant": ResNet18_Variant,
    "resnet18_se_variant": ResNet18_SE_Variant,
    "resnet34": ResNet34,
}


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
    if not name or not isinstance(name, str):
        raise ValueError("`--module` must be a non-empty module path, e.g. 'layer3.0'.")

    cur = model
    for part in name.split("."):
        if part == "":
            raise ValueError(f"Invalid module path '{name}'.")
        if part.isdigit():
            idx = int(part)
            try:
                cur = cur[idx]
            except Exception as err:
                raise ValueError(
                    f"Invalid module path '{name}': index '{idx}' is not available."
                ) from err
        else:
            if not hasattr(cur, part):
                raise ValueError(
                    f"Invalid module path '{name}': attribute '{part}' was not found."
                )
            cur = getattr(cur, part)

    if not isinstance(cur, nn.Module):
        raise ValueError(
            f"Module path '{name}' does not resolve to a torch.nn.Module."
        )

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
    Run dummy input through model to figure out number of channels
    in the target module.
    """
    module = get_module(model, module_name)
    hook = Hook(module)
    try:
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        _ = model(x)
        act = hook.act
        if act is None:
            raise RuntimeError(
                f"No activations captured for module '{module_name}'."
            )
        if act.dim() < 2:
            raise ValueError(
                f"Module '{module_name}' output must have at least 2 dimensions. Got shape {tuple(act.shape)}."
            )
        return int(act.shape[1])
    finally:
        hook.close()


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
    try:
        if init_img is None:
            param = torch.randn(
                1,
                3,
                image_size,
                image_size,
                device=device,
                requires_grad=True,
            )
        else:
            eps = 1e-6
            img0 = init_img.clamp(eps, 1 - eps)
            param = torch.log(img0 / (1 - img0)).detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([param], lr=lr)

        for step in range(steps):
            opt.zero_grad()

            # convert param to image using sigmoid
            img = torch.sigmoid(param)

            # random spatial jitter
            if jitter_px > 0:
                dx = random.randint(-jitter_px, jitter_px)
                dy = random.randint(-jitter_px, jitter_px)
                img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))

            # training/inference use tensors without normalization
            x = img
            _ = model(x)

            act = hook.act
            if act is None:
                raise RuntimeError(
                    f"No activations captured for module '{module_name}'."
                )
            if act.dim() < 2:
                raise ValueError(
                    f"Module '{module_name}' output must have at least 2 dimensions. Got shape {tuple(act.shape)}."
                )
            if channel_idx < 0 or channel_idx >= act.shape[1]:
                raise ValueError(
                    f"Channel index {channel_idx} is out of range for module '{module_name}' "
                    f"(valid range: 0-{act.shape[1] - 1})."
                )

            # select desired channel
            act_ch = act[:, channel_idx]

            # compute objective
            obj = topk_objective(act_ch, topk)

            # regularization losses
            l2 = img.pow(2).mean()
            tv = total_variation(img)

            # total loss (negative because we maximize)
            loss = -obj + l2_weight * l2 + tv_weight * tv

            loss.backward()
            opt.step()

            # blur occasionally
            if blur_every > 0 and (step + 1) % blur_every == 0:
                with torch.no_grad():
                    img2 = torch.sigmoid(param)
                    img2 = F.avg_pool2d(img2, 3, 1, 1)
                    eps = 1e-6
                    img2 = img2.clamp(eps, 1 - eps)
                    param.data = torch.log(img2 / (1 - img2))

        return torch.sigmoid(param).detach()
    finally:
        hook.close()


def parse_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized not in MODEL_REGISTRY:
        options = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {options}"
        )
    return normalized


def parse_channels(channels_arg: str, num_channels: int) -> list[int]:
    if channels_arg.strip().lower() == "all":
        return list(range(num_channels))

    raw_parts = [part.strip() for part in channels_arg.split(",")]
    if not raw_parts or any(part == "" for part in raw_parts):
        raise ValueError(
            "Invalid `--channels` value. Use comma-separated integers (e.g. '0,2,4') or 'all'."
        )

    channels = []
    for part in raw_parts:
        try:
            ch = int(part)
        except ValueError as err:
            raise ValueError(
                f"Invalid channel '{part}'. Channels must be integers or 'all'."
            ) from err
        if ch < 0:
            raise ValueError("Channels must be non-negative.")
        if ch >= num_channels:
            raise ValueError(
                f"Channel {ch} is out of range for selected module (valid range: 0-{num_channels - 1})."
            )
        channels.append(ch)

    if not channels:
        raise ValueError("At least one channel must be provided.")

    return channels


def load_checkpoint(model: nn.Module, ckpt_path: Path, model_name: str) -> None:
    if not ckpt_path.exists() or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is missing 'model_state'."
        )

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as err:
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' is not compatible with model '{model_name}'. "
            "Please pass a matching `--model` for this checkpoint."
        ) from err


# -----------------------------
# main
# -----------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Activation maximization for supported ResNet architectures."
    )

    # currently only important args are exposed, but feel free to add more if you want to experiment with different things
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18_variant",
        help="Model name (resnet18, resnet18_se, resnet18_variant, resnet18_se_variant, resnet34)",
    )
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--module", type=str, default="layer3.0")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--init_img", type=str, default="")
    parser.add_argument("--channels", type=str, default="0,1,2")
    parser.add_argument("--topk", type=int, default=100)

    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("`--topk` must be a positive integer.")
    if args.image_size <= 0:
        parser.error("`--image_size` must be a positive integer.")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists() or not ckpt_path.is_file():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    init_img_path = None
    if args.init_img:
        init_img_path = Path(args.init_img)
        if not init_img_path.exists() or not init_img_path.is_file():
            parser.error(f"Initial image not found: {init_img_path}")

    outdir_path = Path(args.outdir)
    try:
        outdir_path.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        parser.error(f"Failed to create output directory '{outdir_path}': {err}")

    try:
        model_name = parse_model_name(args.model)
    except ValueError as err:
        parser.error(str(err))
    model = MODEL_REGISTRY[model_name](num_classes=6)

    try:
        load_checkpoint(model, ckpt_path, model_name)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        parser.error(str(err))

    # choose gpu if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device).eval()
    print(f"Loaded checkpoint for model: {model_name}")

    # optional init image
    init_img = None
    if init_img_path is not None:
        try:
            init_img = load_init_image(init_img_path, args.image_size, device)
        except Exception as err:
            parser.error(f"Failed to load `--init_img`: {err}")

    try:
        num_channels = infer_channels(
            model,
            args.module,
            args.image_size,
            device
        )
    except (ValueError, RuntimeError) as err:
        parser.error(str(err))

    try:
        channel_list = parse_channels(args.channels, num_channels)
    except ValueError as err:
        parser.error(str(err))

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
            str(outdir_path),
            f"{args.module.replace('.','_')}_ch{ch}.png"
        )

        save_image(img_np, path,
                   f"{args.module} ch{ch}")

    print("done")


if __name__ == "__main__":
    main()
