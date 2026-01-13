'''
Inference Script for iterating over a folder of images and writing the corresponding classification 
scores to a CSV file.

Quickstart
----------
1) Create and activate a virtual environment

macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate

Windows (PowerShell):
    py -m venv venv
    .\\venv\\Scripts\\Activate.ps1

2) Install dependencies
    pip install -r requirements.txt

3) Run
    python demo/demo_csv.py --folder_path "PATH/TO/IMAGE/FOLDER" --model_path "PATH/TO/CHECKPOINT.pt"

Optional:
    python demo/demo_csv.py --folder_path "..." --model_path "..." --output_csv "PATH/TO/CSV_NAME"
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

from models import ResNet18_SE_Variant

EMOTIONS = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]


def load_model(model_weights,device): 
    model = ResNet18_SE_Variant(num_classes=6)
    checkpoint = torch.load(model_weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model 

def classify_image(model, image_path,device):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(img_tensor)
        preds = torch.softmax(out, dim=1)

    return preds.cpu().numpy()[0]

def write_csv(folder_path, model_weights,output_csv):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using Device: {device}")

    model = load_model(model_weights,device)
    print(f"Using model {model.__class__.__name__}")

    folder = Path(folder_path)
    image_files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

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
        description='Classifying Emotions in images and write probabilites to CSV file'
    )
    parser.add_argument(
        '--folder_path',
        type=str,
        required=True,
        help='Path to folder containing images',
    )

    parser.add_argument(
        "--model_path",
        type = str,
        default= "ResNet18_SE_Variant_best.pt",
        help = "Path to model chekpoint (.pt)",
    )
    parser.add_argument(
        "--output_csv",
        type = str,
        default = str(Path("model_metrics") / "inference_csv" / "predictions.csv"),
        help = "Output CSV file name",
    )
    
    args = parser.parse_args()
    

    try:
        write_csv(
            folder_path=args.folder_path,
            model_weights=args.model_path,
            output_csv=args.output_csv,
        )
    except Exception as e:
        print(f"Fehler: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()