"""
Generate saliency heatmaps for all images in a folder using GradCAM or other CAM methods.
Uses pytorch-grad-cam. Only use trusted checkpoints (weights_only is false).

Supported models: resnet18, resnet18_se, resnet18_variant, resnet18_se_variant, resnet34

Usage:
    python demo/demo_cam.py --folder_path "PATH/TO/IMAGE/FOLDER" --model resnet18_variant --model_path "PATH/TO/CHECKPOINT.pt"
    python demo/demo_cam.py --folder_path "..." --model "..." --model_path "..." --cam gradcam --target_layer 4 --output_path "PATH/TO/OUTPUT"
"""

import sys
import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 
from pytorch_grad_cam import GradCAM,GradCAMPlusPlus,ScoreCAM,EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18, ResNet18_SE, ResNet18_Variant, ResNet18_SE_Variant, ResNet34

EMOTIONS = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
])


def load_model(model_name,model_weights,device):

    model_dict = {
        'resnet18': ResNet18,
        'resnet18_se': ResNet18_SE,
        'resnet18_variant': ResNet18_Variant,
        'resnet18_se_variant': ResNet18_SE_Variant,
        'resnet34': ResNet34
    }

    if model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    model = model_dict[model_name.lower()](num_classes=6)
    checkpoint = torch.load(model_weights, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError as e:
        raise RuntimeError(
            f"Model weights not compatible with model '{model_name}'.\n{e}"
        )
    model.to(device)
    model.eval()
    return model


def do_grad(model_name, model_weights, cam_method, layer, folder_path, output_path):

    cam_dict = {
        'gradcam': GradCAM,
        'plusplus': GradCAMPlusPlus,
        'score': ScoreCAM,
        'eigen': EigenCAM,
        'layer': LayerCAM
    }

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = load_model(model_name, model_weights, device)

    layer_dict = {
        '1': model.layer1,
        '2': model.layer2,
        '3': model.layer3,
        '4': model.layer4
    }

    target_layers = [layer_dict[layer][-1]]

    if cam_method.lower() not in cam_dict:
        raise ValueError(f"Unknown CAM method: {cam_method}. Available: {list(cam_dict.keys())}")
    
    cam = cam_dict[cam_method.lower()](
        model = model,
        target_layers= target_layers
    )

    output_path = Path(output_path)
    heatmap_dir = output_path 
    heatmap_dir.mkdir(parents = True, exist_ok=True )

    image_paths = [p for p in Path(folder_path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        print("Error: No .jpg/.jpeg/.png images found in folder.", file = sys.stderr)
        sys.exit(1)

    for img_path in image_paths:
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        input_tensor = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        targets = [ClassifierOutputTarget(pred_idx)] 

        grayscale_cam = cam(
            input_tensor = input_tensor,
            targets = targets
        )[0]

        rgb_img = np.array(img.resize((64, 64))) / 255.0
        cam_image = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        out_name = (
            f"{img_path.stem}_"
            f"{EMOTIONS[pred_idx]}_"
            f"{confidence:.2f}.png"
        )

        out_path = heatmap_dir / out_name
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        )

        print(f"{img_path.name} → {out_name}")
    print(f"\n Grad-CAM completed. Heatmaps in: {heatmap_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Doing Gradcam on images in folder'
    )
    parser.add_argument(
        '--folder_path',
        type=str,
        required=True,
        help='Path to folder containing images',
    )

    parser.add_argument(
        "--model",
        type = str,
        default='resnet18_variant',
        choices=['resnet18', 'resnet18_se', 'resnet18_variant', 'resnet18_se_variant', 'resnet34'],
        help = "Model selection",
    )

    parser.add_argument(
        "--model_path",
        type = str,
        required = True,
        help = "Path to model chekpoint (.pt)",
    )

    parser.add_argument(
        "--cam",
        type = str,
        default = "gradcam",
        choices = ["gradcam", "plusplus", "eigen", "score", "layer"],
        help = "Choose CAM method"
    )

    parser.add_argument(
        "--output_path",
        type = str,
        default = str(PROJECT_ROOT/"demo"/"heatmaps"),
        help = "Choose output folder",
    )

    parser.add_argument(
        "--target_layer",
        type = str,
        default = '4',
        choices = ['1', '2', '3', '4'],
        help = "Choose target layers"
    )
    
    args = parser.parse_args()

    try:
        do_grad(
            model_name=args.model,
            model_weights=args.model_path,
            cam_method=args.cam,
            layer=args.target_layer,
            folder_path=args.folder_path, 
            output_path=args.output_path
        )
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()