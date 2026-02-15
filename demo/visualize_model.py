"""
Generate detailed architecture diagrams for supported ResNet18 variants.

Supported models:
    - resnet18
    - ResNet18_SE
    - ResNet18_SE_Variant
    - ResNet18_Variant

Usage:
    python demo/visualize_model.py --model resnet18
    python demo/visualize_model.py --model ResNet18_SE
"""

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


MODEL_CONFIGS = {
    "resnet18": {
        "display_name": "ResNet18",
        "filename_prefix": "resnet18",
        "is_variant": False,
        "has_se": False,
    },
    "resnet18_se": {
        "display_name": "ResNet18_SE",
        "filename_prefix": "resnet18_se",
        "is_variant": False,
        "has_se": True,
    },
    "resnet18_se_variant": {
        "display_name": "ResNet18_SE_Variant",
        "filename_prefix": "resnet18_se_variant",
        "is_variant": True,
        "has_se": True,
    },
    "resnet18_variant": {
        "display_name": "ResNet18_Variant",
        "filename_prefix": "resnet18_variant",
        "is_variant": True,
        "has_se": False,
    },
}

SUPPORTED_MODELS = (
    "resnet18",
    "ResNet18_SE",
    "ResNet18_SE_Variant",
    "ResNet18_Variant",
)
DEFAULT_NUM_CLASSES = 6


def resolve_model_config(model_name: str) -> dict:
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized not in MODEL_CONFIGS:
        options = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {options}")
    return MODEL_CONFIGS[normalized]


def draw_detailed_diagram(model_config: dict) -> str:
    model_name = model_config["display_name"]
    is_variant = model_config["is_variant"]
    has_se = model_config["has_se"]

    fig, ax = plt.subplots(1, 1, figsize=(19, 9))
    ax.axis("off")

    colors = {
        "input": "#E8F5E9",
        "conv": "#BBDEFB",
        "bn": "#FFF9C4",
        "relu": "#FFCCBC",
        "pool": "#E1BEE7",
        "fc": "#FFCDD2",
        "add": "#B2DFDB",
        "se": "#B2EBF2",
        "block_bg": "#F5F5F5",
        "skip": "#FF9800",
    }

    def draw_box(x, y, width, height, label, sublabel, color, fontsize=8):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height / 2 + (0.08 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )
        if sublabel:
            ax.text(
                x + width / 2,
                y + height / 2 - 0.19,
                sublabel,
                ha="center",
                va="center",
                fontsize=6,
                color="gray",
            )

    def draw_arrow(x1, y1, x2, y2, color="black"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
        )

    if is_variant:
        conv1_channels = 32
        layer_info = [
            ("layer1", "64x64x32", "s=1"),
            ("layer2", "32x32x64", "s=2"),
            ("layer3", "16x16x128", "s=2"),
            ("layer4", "16x16x256", "s=1"),
        ]
        pool_channels = 256
        fc_input = 256
    else:
        conv1_channels = 64
        layer_info = [
            ("layer1", "64x64x64", "s=1"),
            ("layer2", "32x32x128", "s=2"),
            ("layer3", "16x16x256", "s=2"),
            ("layer4", "8x8x512", "s=2"),
        ]
        pool_channels = 512
        fc_input = 512

    ax.text(10.5, 8.4, f"{model_name} Architecture Overview", ha="center", fontsize=14, fontweight="bold")

    x = 0.6
    y = 6.4
    draw_box(x, y, 1.0, 1.0, "Input", "64x64x3", colors["input"])
    draw_arrow(x + 1.05, y + 0.5, x + 1.45, y + 0.5)
    x += 1.5

    draw_box(x, y, 1.25, 1.0, "Conv1", f"3x3, {conv1_channels}, s=1", colors["conv"])
    draw_arrow(x + 1.3, y + 0.5, x + 1.7, y + 0.5)
    x += 1.75

    draw_box(x, y, 0.8, 1.0, "BN", "", colors["bn"])
    draw_arrow(x + 0.85, y + 0.5, x + 1.25, y + 0.5)
    x += 1.3

    draw_box(x, y, 0.8, 1.0, "ReLU", "", colors["relu"])
    draw_arrow(x + 0.85, y + 0.5, x + 1.25, y + 0.5)
    x += 1.3

    for idx, (name, out_size, stride) in enumerate(layer_info):
        draw_box(x, y - 0.1, 2.0, 1.2, name, f"{out_size}\n{stride}", colors["block_bg"])
        x += 2.2
        if idx < len(layer_info) - 1:
            draw_arrow(x - 0.15, y + 0.5, x + 0.25, y + 0.5)
            x += 0.3

    draw_arrow(x - 0.15, y + 0.5, x + 0.25, y + 0.5)
    x += 0.3
    draw_box(x, y, 1.2, 1.0, "AvgPool", f"1x1x{pool_channels}", colors["pool"])
    x += 1.35

    if is_variant:
        draw_arrow(x - 0.15, y + 0.5, x + 0.25, y + 0.5)
        x += 0.3
        draw_box(x, y, 1.0, 1.0, "Drop", "p=0.3", "#FFECB3")
        x += 1.15

    draw_arrow(x - 0.15, y + 0.5, x + 0.25, y + 0.5)
    x += 0.3
    draw_box(x, y, 1.0, 1.0, "FC", f"{fc_input}->{DEFAULT_NUM_CLASSES}", colors["fc"])
    draw_arrow(x + 1.05, y + 0.5, x + 1.45, y + 0.5)
    x += 1.5
    draw_box(x, y, 1.0, 1.0, "Output", f"{DEFAULT_NUM_CLASSES} cls", colors["input"])

    ax.text(10.5, 4.5, "BasicBlock Internal Structure (x2 per Layer)", ha="center", fontsize=12, fontweight="bold")

    block_x, block_y = 3.0, 0.3
    block_w, block_h = 15.0, 3.9
    block_bg = FancyBboxPatch(
        (block_x, block_y),
        block_w,
        block_h,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor="#F8F8F8",
        edgecolor="#333333",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(block_bg)

    block_center_x = block_x + block_w / 2
    main_flow_width = 11.0 + (1.7 if has_se else 0.0)
    main_flow_left = block_center_x - (main_flow_width / 2)

    path_x = main_flow_left + 0.25
    path_y = 2.5
    ax.text(path_x - 0.25, path_y + 0.1, "x", fontsize=10, fontweight="bold")
    fork_x = path_x - 0.2
    draw_arrow(path_x - 0.05, path_y + 0.1, path_x + 0.35, path_y + 0.1)

    path_x += 0.45
    draw_box(path_x, path_y - 0.3, 1.5, 0.8, "Conv2d", "3x3", colors["conv"], fontsize=7)
    draw_arrow(path_x + 1.55, path_y + 0.1, path_x + 1.95, path_y + 0.1)
    path_x += 2.0

    draw_box(path_x, path_y - 0.3, 0.65, 0.8, "BN", "", colors["bn"], fontsize=7)
    draw_arrow(path_x + 0.7, path_y + 0.1, path_x + 1.1, path_y + 0.1)
    path_x += 1.15

    draw_box(path_x, path_y - 0.3, 0.85, 0.8, "ReLU", "", colors["relu"], fontsize=7)
    draw_arrow(path_x + 0.9, path_y + 0.1, path_x + 1.3, path_y + 0.1)
    path_x += 1.35

    draw_box(path_x, path_y - 0.3, 1.5, 0.8, "Conv2d", "3x3", colors["conv"], fontsize=7)
    draw_arrow(path_x + 1.55, path_y + 0.1, path_x + 1.95, path_y + 0.1)
    path_x += 2.0

    draw_box(path_x, path_y - 0.3, 0.65, 0.8, "BN", "", colors["bn"], fontsize=7)
    draw_arrow(path_x + 0.7, path_y + 0.1, path_x + 1.1, path_y + 0.1)
    path_x += 1.15

    if has_se:
        draw_box(path_x, path_y - 0.3, 1.2, 0.8, "SE Block", "Squeeze-Excite", colors["se"], fontsize=7)
        draw_arrow(path_x + 1.25, path_y + 0.1, path_x + 1.65, path_y + 0.1)
        path_x += 1.7

    add_x = path_x + 0.35
    add_y = path_y + 0.1
    add_circle = plt.Circle((add_x, add_y), 0.3, facecolor=colors["add"], edgecolor="black", linewidth=1.4)
    ax.add_patch(add_circle)
    ax.text(add_x, add_y, "+", ha="center", va="center", fontsize=12, fontweight="bold")
    draw_arrow(add_x + 0.35, add_y, add_x + 0.85, add_y)
    draw_box(add_x + 0.9, path_y - 0.3, 0.85, 0.8, "ReLU", "", colors["relu"], fontsize=7)
    draw_arrow(add_x + 1.8, add_y, add_x + 2.2, add_y)
    ax.text(add_x + 2.3, add_y, "out", fontsize=10, fontweight="bold", va="center")

    skip_box_width = 2.2
    skip_box_height = 0.8
    skip_vertical_gap = 0.35
    main_path_bottom = path_y - 0.3
    skip_y = main_path_bottom - skip_vertical_gap - skip_box_height
    skip_mid_y = skip_y + skip_box_height / 2
    skip_box_x = (fork_x + add_x) / 2 - skip_box_width / 2

    skip_start_y = path_y - 0.05
    ax.plot([fork_x, fork_x], [skip_start_y, skip_mid_y], color=colors["skip"], linewidth=2)
    draw_arrow(fork_x, skip_mid_y, skip_box_x, skip_mid_y, color=colors["skip"])
    draw_box(skip_box_x, skip_y, skip_box_width, skip_box_height, "Identity / 1x1 Conv", "", "#FFE0B2", fontsize=7)
    draw_arrow(skip_box_x + skip_box_width, skip_mid_y, add_x, skip_mid_y, color=colors["skip"])
    draw_arrow(add_x, skip_mid_y, add_x, add_y - 0.28, color=colors["skip"])
    ax.text((fork_x + add_x) / 2, skip_y - 0.3, "Shortcut (if dims change)", fontsize=7, color="#E65100", ha="center")

    legend_items = [
        (colors["conv"], "Convolution"),
        (colors["bn"], "Batch Norm"),
        (colors["relu"], "ReLU"),
        (colors["add"], "Add"),
        (colors["pool"], "Pooling"),
        (colors["fc"], "FC"),
    ]
    if has_se:
        legend_items.insert(4, (colors["se"], "SE Block"))

    legend_x = 2.0
    legend_y = -1.0
    for color, label in legend_items:
        draw_box(legend_x, legend_y, 0.35, 0.35, "", "", color)
        ax.text(legend_x + 0.5, legend_y + 0.17, label, fontsize=7, va="center")
        legend_x += 2.4

    ax.plot([legend_x, legend_x + 0.6], [legend_y + 0.17, legend_y + 0.17], color=colors["skip"], linewidth=2)
    ax.text(legend_x + 0.75, legend_y + 0.17, "Skip", fontsize=7, va="center")

    ax.set_xlim(0.0, 22.4)
    ax.set_ylim(-1.8, 9.1)

    filename = f"{model_config['filename_prefix']}_detailed_diagram.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detailed architecture diagrams for supported ResNet18 variants."
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name (resnet18, ResNet18_SE, ResNet18_SE_Variant, ResNet18_Variant)",
    )
    args = parser.parse_args()

    try:
        model_config = resolve_model_config(args.model)
    except ValueError as err:
        parser.error(str(err))

    filename = draw_detailed_diagram(model_config)
    print(f"Saved detailed block diagram to: {filename}")


if __name__ == "__main__":
    main()
