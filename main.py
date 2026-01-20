import argparse
from models import ResNet18, ResNet18_SE, ResNet18_SE_Variant
from training.ResNetTrainer import ResNetTrainer


def get_model(model_name: str):
    """Load the desired model based on the name."""
    models = {
        'resnet18': ResNet18,
        'resnet18_se': ResNet18_SE,
        'resnet18_se_variant': ResNet18_SE_Variant,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()]()


def main():
    parser = argparse.ArgumentParser(
        description='Train Facial Expression Recognition Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy examples (still work)
  python main.py --model resnet18_se_variant --epochs 50 --lr 0.001
  python main.py --model resnet18 --batch-size 64 --use-adamw
  python main.py --model resnet18_se --epochs 100 --use-scheduler --use-label-smoothing
  python main.py --model resnet18 --use-class-weights --use-label-smoothing
  
  # Dataset selection with "all" shortcut
  python main.py --model resnet18 --train-datasets all --val-datasets all
  python main.py --model resnet18 --train-datasets all --val-datasets raf_db
  python main.py --model resnet18 --train-datasets affectnet fer2013 --val-datasets human_emotions
  
  # Class limiting (helps with class imbalance)
  python main.py --model resnet18 --train-datasets all --val-datasets all --class-limit 50000
  python main.py --model resnet18 --train-datasets all --val-datasets raf_db --class-limit 30000
  
  # Combined examples
  python main.py --model resnet18 --train-datasets all --val-datasets all --class-limit 40000 --use-adamw --use-scheduler
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18_se_variant',
        choices=['resnet18', 'resnet18_se', 'resnet18_se_variant'],
        help='Select the model (default: resnet18_se_variant)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=32,
        help='Number of training epochs (default: 32)'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='Weight decay (default: 0.0001)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--cm-every',
        type=int,
        default=1,
        help='Save confusion matrix every N epochs (default: 5, set to 1 for every epoch)'
    )
    
    parser.add_argument(
        '--use-adamw',
        action='store_true',
        help='Use AdamW instead of Adam optimizer'
    )
    
    parser.add_argument(
        '--use-scheduler',
        action='store_true',
        help='Use learning rate scheduler (OneCycleLR)'
    )
    
    parser.add_argument(
        '--use-label-smoothing',
        action='store_true',
        help='Use label smoothing in CrossEntropyLoss'
    )
    
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use class weights in CrossEntropyLoss to handle class imbalance'
    )
    
    parser.add_argument(
        '--no-sampler',
        action='store_true',
        help='Disable WeightedRandomSampler for class balance'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='full',
        choices=['full', 'sample'],
        help='Select the dataset (default: full) - Legacy option'
    )
    
    parser.add_argument(
        '--train-datasets',
        type=str,
        nargs='+',
        choices=['all', 'affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db'],
        help='Training datasets (space-separated). Use "all" for all 5 datasets'
    )
    
    parser.add_argument(
        '--val-datasets',
        type=str,
        nargs='+',
        choices=['all', 'affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db'],
        help='Validation datasets (space-separated). Use "all" for all 5 datasets'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default = 5,
        help = "Set patience (in epochs) for Early Stopping."
    )
    
    parser.add_argument(
        '--class-limit',
        type=int,
        default=None,
        help='Limit maximum number of samples per class (e.g., 50000). Helps with class imbalance.'
    )
    
    args = parser.parse_args()
    
    # Validate dataset arguments
    if (args.train_datasets is None) != (args.val_datasets is None):
        parser.error("Both --train-datasets and --val-datasets must be specified together or not at all.")
    
    # Handle 'all' shortcut
    all_datasets = ['affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db']
    if args.train_datasets and 'all' in args.train_datasets:
        args.train_datasets = all_datasets
    if args.val_datasets and 'all' in args.val_datasets:
        args.val_datasets = all_datasets
    
    # Load model
    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    
    """
    # This output is already generated by the trainer itself. 
    # Create trainer with arguments
    print(f"\nTrainer Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  AdamW: {args.use_adamw}")
    print(f"  Scheduler: {args.use_scheduler}")
    print(f"  Label Smoothing: {args.use_label_smoothing}")
    print(f"  Class Weights: {args.use_class_weights}")
    print(f"  Confusion Matrix every: {args.cm_every} epochs")
    print()
    """
    
    trainer = ResNetTrainer(
        model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dataset=args.dataset,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        cm_every=args.cm_every,
        use_adamw=args.use_adamw,
        use_scheduler=args.use_scheduler,
        use_label_smoothing=args.use_label_smoothing,
        use_class_weights=args.use_class_weights,
        class_limit=args.class_limit,
        early_stopping_patience=args.patience   
    )
    
    trainer.train()


if __name__ == '__main__':
    main()

