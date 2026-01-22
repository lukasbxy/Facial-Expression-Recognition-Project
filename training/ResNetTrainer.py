from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
import sys
from training.create_cm import create_cm
import csv
from torchmetrics.classification import MulticlassF1Score

from training.load_data import get_dataloaders
from training.early_stopping import EarlyStopping

class ResNetTrainer:
    
    def __init__(self, 
                 model,
                 num_epochs: int = 32, 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001,
                 train_datasets=None,
                 val_datasets=None,
                 cm_every: int = 1,
                 use_adamw: bool = False,
                 use_scheduler: bool = False,
                 use_label_smoothing: bool = False,
                 use_class_weights: bool = False,
                 class_limit: int = None,
                 best_model_filename: str = "best.pt",
                 last_model_filename: str = "last.pt",
                 early_stopping_patience: int = 5):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.cm_every = cm_every
        self.use_scheduler = use_scheduler
        self.use_class_weights = use_class_weights
        self.class_limit = class_limit
        self.best_model_filename = best_model_filename
        self.last_model_filename = last_model_filename
        self.early_stopping_patience = early_stopping_patience
        
        # Automatically construct filepath based on model class name
        # Store model_name in self for use in logging and filenames
        self.model_name = model.__class__.__name__
        
        # Generate timestamp for this training session
        self.timestamp = datetime.now().strftime("%d.%m.%y_%H.%M.%S")
        
        # Setup logging
        self._setup_checkpoints_logging()
        
        # Create metrics file
        self._setup_metrics()
        
        # Set Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Using Device: {self.device}")
        
        # Model
        self.model = model
        self.model = self.model.to(self.device)
        
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device = self.device.type, enabled = self.use_amp)
        
        # Dataloaders
        self.train_loader, self.val_loader, self.class_names = get_dataloaders(
            train_datasets=self.train_datasets, 
            val_datasets=self.val_datasets,
            class_limit=self.class_limit
        )
        
        # Compute class weights if needed
        if use_class_weights:
            # Automatically compute class weights from training dataset
            # Gather targets from all sub-datasets (handles ConcatDataset and Subset)
            all_targets = []
            train_dataset = self.train_loader.dataset
            if hasattr(train_dataset, 'datasets'):
                # ConcatDataset case
                for ds in train_dataset.datasets:
                    if hasattr(ds, 'targets'):
                        all_targets.extend(ds.targets)
                    elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'targets'):
                        # Subset case
                        indices = ds.indices
                        original_targets = ds.dataset.targets
                        all_targets.extend([original_targets[i] for i in indices])
            elif hasattr(train_dataset, 'targets'):
                all_targets = list(train_dataset.targets)
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
                # Single Subset case
                indices = train_dataset.indices
                original_targets = train_dataset.dataset.targets
                all_targets = [original_targets[i] for i in indices]
            
            targets = np.array(all_targets)
            class_counts = np.bincount(targets)
            # Compute inverse frequency weights and normalize
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = None
            
        
        # Optimizer
        # Adam or AdamW
        if use_adamw:    
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr = self.learning_rate,
                weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr = self.learning_rate,
                weight_decay = self.weight_decay)
        
        
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=3e-4,
                epochs=num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,         
                div_factor=10.0,        
                final_div_factor=100.0  
        )
            
        if use_label_smoothing:
            if use_class_weights and class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            if use_class_weights and class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
                
        # Setup F1 Score
        self.num_classes = len(self.class_names)
        self.val_f1_macro = MulticlassF1Score(num_classes=self.num_classes, average="macro").to(self.device)
        
        self.logger.info("Trainer initialized.")
        
    def _setup_checkpoints_logging(self):
        """Setup logging and file_system"""
        # Create runs directory if it doesn't exist
        log_dir = Path("runs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create folder for model run
        self.run_dir = log_dir / self.model_name / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        log_file = self.run_dir / "train.log"
        
        # Configure logger
        self.logger = logging.getLogger(f"{self.model_name}_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # setup checkpoint folders
        self.checkpoints_path = self.run_dir / "checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Logging initialized. Run directory: {self.run_dir}")
        
    def _setup_metrics(self):
        self.metrics_path = self.run_dir / "metrics.csv"
        self._metrics_file = self.metrics_path.open("a", newline="")
        self._metrics_writer = None
        
    def _log_metrics(self, metrics: dict):
        if self._metrics_writer is None:
            self._metrics_writer = csv.DictWriter(
                self._metrics_file,
                fieldnames=list(metrics.keys())
            )
            self._metrics_writer.writeheader()
        self._metrics_writer.writerow(metrics)
        self._metrics_file.flush()
        
    def _get_config(self):
        """Print out selected config of trainer for logging purposes!"""
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_datasets": self.train_datasets,
            "val_datasets": self.val_datasets,
            "cm_every": self.cm_every,
            "use_scheduler": self.use_scheduler,
            "use_class_weights": self.use_class_weights,
            "class_limit": self.class_limit,
            "best_model_filename": self.best_model_filename,
            "last_model_filename": self.last_model_filename,
            "device": str(self.device),
            "model.class": self.model_name,
            "scheduler": self.scheduler.__class__.__name__ if self.use_scheduler else None,
            "optimizer": self.optimizer.__class__.__name__,
            "early_stopping_patience": str(self.early_stopping_patience)
        }
        
    def train_one_epoch(self, epoch):
        """
        Train model for one epoch.
        """
        
        self.model.train()
        
        loss_total = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]"):
            # Load images to GPU/CPU
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            if self.use_scheduler:
                self.scheduler.step()
            
            predicted = outputs.argmax(dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_total += loss.item() * labels.size(0)
            
        loss_avg = loss_total / total
        accuracy = 100 * correct / total
        return loss_avg, accuracy
    
    
    def validate(self, epoch):
        """Validate module on validation data"""
        self.val_f1_macro.reset()
        self.model.eval()
        loss_total, correct, total = 0.0, 0, 0
        
        do_cm = (self.cm_every > 0) and ((epoch+1) % self.cm_every == 0)
        all_preds = [] 
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
                # Load images to GPU/CPU
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                predicted = outputs.argmax(dim = 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_total += loss.item() * labels.size(0)
                self.val_f1_macro.update(predicted, labels) # Update F1 scoring

                if do_cm:
                    all_preds.append(predicted.detach().cpu())
                    all_labels.append(labels.detach().cpu())
                
            if do_cm and len(all_preds) > 0:
                labels_np = torch.cat(all_labels).numpy()
                preds_np  = torch.cat(all_preds).numpy()
                
                # Save confusion matrics
                cm_path = create_cm(labels=labels_np,
                                 preds=preds_np,
                                 class_names = self.class_names,
                                 epoch = epoch,
                                 model_name = self.model_name,
                                 timestamp = self.timestamp,
                                 out_dir = self.run_dir / "confusion_matrices")
                
                self.logger.info(f"Confusion matrix saved at {cm_path}")
            
        loss_avg = loss_total / total
        accuracy = 100 * correct / total
        val_f1_macro = float(self.val_f1_macro.compute().detach().cpu())
        return loss_avg, accuracy, val_f1_macro
    
    
    def _print_epoch_summary(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_f1_macro, current_lr):
        """Print formatted epoch summary"""
        self.logger.info("─" * 66)
        self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
        self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        self.logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        self.logger.info(f"  LR: {current_lr:.6f}")
        self.logger.info("─" * 66)
    
    def save_model(self, path: Path, model, optimizer, epoch: int, best_val_acc: float):
        """Save model to specified path with extra info incl. optimizer and best val acc"""
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            path,
        )    
        self.logger.info(f"Saved best model (val_acc: {best_val_acc:.2f}%) to {path}")
        

    def train(self):
        """Main training loop"""
        self.logger.info("==== Trainer Configuration: ====")
        for k, v in self._get_config().items():
            self.logger.info(f" {k}: {v}")
        self.logger.info("================================")
        
        self.logger.info(f"Beginning training for {self.model_name} for {self.num_epochs} epochs.")
        self.logger.info("-" * 60)
        
        # Store best accuracy during run to determine when to save model.
        best_val_acc = -1.0
        
        # Initialize Early Stopping
        early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=0.001, mode='max')
        
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            val_loss, val_accuracy, val_f1_macro = self.validate(epoch)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self._print_epoch_summary(epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_f1_macro, current_lr)
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model(self.checkpoints_path / self.best_model_filename, self.model, self.optimizer, epoch, best_val_acc)
            
            self.save_model(self.checkpoints_path / self.last_model_filename, self.model, self.optimizer, epoch, best_val_acc)
            
            # Log metrics in run
            self._log_metrics({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "lr": self.optimizer.param_groups[0]["lr"],
                "f1_macro": val_f1_macro
            })
            
            # Check Early Stopping
            if early_stopping.check(val_accuracy):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                self.logger.info(f"No improvement in validation accuracy for {early_stopping.patience} epochs.")
                self.logger.info(f"Best validation accuracy: {early_stopping.best_value:.2f}%")
                self.logger.info(f"{'='*60}")
                break
                
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training complete.")