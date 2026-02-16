r'''
Inference Script for iterating over a folder of images and writing the corresponding classification 
scores to a CSV file.

How to use:
1. Change into project directory

2. Create and activate a virtual environment

macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate
Windows (PowerShell):
    python -m venv venv
    .\venv\Scripts\Activate.ps1

3. Install dependencies
    pip install -r requirements.txt

4. Run
    python demo/demo_csv.py --folder_path "PATH/TO/IMAGE/FOLDER" --model resnet18_se_variant --model_path "PATH/TO/CHECKPOINT.pt"

Optional:
    python demo/demo_csv.py --folder_path "..." --model "..." --model_path "..." --output_csv "PATH/TO/CSV_NAME"
'''



import sys
import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import ResNet18, ResNet18_SE, ResNet18_SE_Variant, ResNet34, CCT

EMOTIONS = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_model(model_name, model_weights, device):
    model_dict = {
        'resnet18': ResNet18,
        'resnet18_se': ResNet18_SE,
        'resnet18_se_variant': ResNet18_SE_Variant,
        'resnet34': ResNet34,
        'cct': CCT
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

def classify_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    img_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(img_tensor)
        preds = torch.softmax(out, dim=1)

    return preds.cpu().numpy()[0]

def write_csv(folder_path, model_name, model_weights, output_csv):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using Device: {device}")

    model = load_model(model_name, model_weights, device)
    print(f"Using model {model.__class__.__name__}")

    folder = Path(folder_path)
    image_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_files:
        print("Error: No images found in input folder.", file=sys.stderr)
        sys.exit(1)

    results = [] 

    for image_path in image_files: 
        preds = classify_image(model=model, image_path= str(image_path), device=device)
        row_data = [str(image_path)] + preds.tolist()
        results.append(row_data)

    columns = ['filepath'] + EMOTIONS
    df = pd.DataFrame(results, columns = columns)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok= True)
    df.to_csv(output_path, index = False, float_format = '%.4f', sep = ";", decimal = ",")
    print(f"CSV saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Classifying Emotions in images and write probabilities to CSV file'
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
        default='resnet18_se_variant',
        choices=['resnet18', 'resnet18_se', 'resnet18_se_variant', 'resnet34', 'cct'],
        help = "Model selection",
    )

    parser.add_argument(
        "--model_path",
        type = str,
        required = True,
        help = "Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_csv",
        type = str,
        default = str(PROJECT_ROOT/"demo" / "csv" / "predictions.csv"),
        help = "Output CSV file name",
    )
    
    args = parser.parse_args()

    input_path = Path(args.folder_path)
    if not input_path.exists() or not input_path.is_dir():
        print("Error: Input folder does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    try:
        write_csv(
            folder_path=args.folder_path,
            model_name=args.model,
            model_weights=args.model_path,
            output_csv=args.output_csv,
        )
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()