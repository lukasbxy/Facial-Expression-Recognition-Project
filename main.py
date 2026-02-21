"""
Script to train facial expression recognition models.

Run `python main.py --help` to see all valid CLI arguments.

How to use (examples):
    python main.py
    python main.py --model resnet18 --epochs 50 --use-scheduler --use-label-smoothing --use-class-weights
    python main.py --model cct --train-datasets all --val-datasets raf_db

It loads the selected model and datasets, picks the matching trainer, and starts training.
"""

import argparse
from models import ResNet18, ResNet18_SE, ResNet18_Variant, ResNet18_SE_Variant, ResNet34, CCT
from training.ResNetTrainer import ResNetTrainer
from training.CCTTrainer import CCTTrainer


def get_model(model_name: str):
    """Return a model instance for the given name."""
    models = {
        'resnet18': ResNet18,
        'resnet18_se': ResNet18_SE,
        'resnet18_se_variant': ResNet18_SE_Variant,
        'resnet34': ResNet34,
        'cct': CCT,
        'resnet18_variant': ResNet18_Variant
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()]()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train facial expression recognition models.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18_variant',
        choices=['resnet18', 'resnet18_se', 'resnet18_se_variant', 'resnet34', 'cct', 'resnet18_variant'],
        help='Select the model (default: resnet18_variant)'
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
        default=None,
        help='Learning rate (default: 0.001 for ResNet, 0.0005 for CCT)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='Weight decay (default: 0.0001 for ResNet, 0.05 for CCT)'
    )
    
    parser.add_argument(
        '--cm-every',
        type=int,
        default=1,
        help='Save confusion matrix every N epochs'
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
    
    parser.add_argument(
        '--disable-weighted-random-sampler', '--disable-wrs',
        action='store_true',
        help='Disable weighted random sampler used to handle class imbalance during training'
    )
    
    parser.add_argument(
        '--weight-power', '--wp',
        type=float,
        default=1.0,
        help='Exponent for class weights when using weighted random sampler (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset arguments
    if (args.train_datasets is None) != (args.val_datasets is None):
        parser.error("Both --train-datasets and --val-datasets must be specified together or not at all.")
    
    # Handle 'all' shortcut and default to all datasets if not specified
    all_datasets = ['affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db']
    if args.train_datasets is None:
        args.train_datasets = all_datasets
    elif 'all' in args.train_datasets:
        args.train_datasets = all_datasets
    if args.val_datasets is None:
        args.val_datasets = all_datasets
    elif 'all' in args.val_datasets:
        args.val_datasets = all_datasets
    
    # Load model
    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    
    # Select trainer based on model type
    if args.model.lower() == 'cct':
        # CCT: Defaults are 0.0005 (lr) and 0.05 (weight_decay)
        trainer = CCTTrainer(
            model,
            num_epochs=args.epochs,
            learning_rate=args.lr if args.lr is not None else 0.0005,
            weight_decay=args.weight_decay if args.weight_decay is not None else 0.05,
            train_datasets=args.train_datasets,
            val_datasets=args.val_datasets,
            cm_every=args.cm_every,
            use_scheduler=args.use_scheduler,
            use_label_smoothing=args.use_label_smoothing,
            use_class_weights=args.use_class_weights,
            class_limit=args.class_limit,
            early_stopping_patience=args.patience,
            use_weighted_sampler=not args.disable_weighted_random_sampler,
            weight_power=args.weight_power
        )
    else:
        # ResNet: Defaults are 0.001 (lr) and 0.0001 (weight_decay)
        trainer = ResNetTrainer(
            model,
            num_epochs=args.epochs,
            learning_rate=args.lr if args.lr is not None else 0.001,
            weight_decay=args.weight_decay if args.weight_decay is not None else 0.0001,
            train_datasets=args.train_datasets,
            val_datasets=args.val_datasets,
            cm_every=args.cm_every,
            use_adamw=args.use_adamw,
            use_scheduler=args.use_scheduler,
            use_label_smoothing=args.use_label_smoothing,
            use_class_weights=args.use_class_weights,
            class_limit=args.class_limit,
            early_stopping_patience=args.patience,
            use_weighted_sampler=not args.disable_weighted_random_sampler,
            weight_power=args.weight_power
        )
    
    trainer.train()


if __name__ == '__main__':
    main()

