"""
Implementierung eines Training-Loops für CCT (models/CCT)
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import logging
import sys
from training.create_cm import create_cm

from training.load_data import get_dataloaders
from training.early_stopping import EarlyStopping

class CCTTrainer:
    
    def __init__(self, 
                 model,
                 num_epochs: int = 30,
                 learning_rate: float = 0.0005,
                 weight_decay: float = 0.05,
                 cm_every: int = 5,
                 use_scheduler: bool = True,
                 use_label_smoothing: bool = True):
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cm_every = cm_every
        self.use_scheduler = use_scheduler
        
        # Store model name and generate timestamp
        self.model_name = model.__class__.__name__
        self.timestamp = datetime.now().strftime("%d.%m.%y_%H.%M")
        
        # Update filepath with timestamp naming scheme
        self.filepath = Path("models") / self.model_name / "checkpoints" / f"{self.timestamp}_Checkpoint_{self.model_name}.pt"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # --- OPTIMIZATION: CUDA Benchmark ---
        torch.backends.cudnn.benchmark = True
        
        # Set Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Using Device: {self.device}")
        
        # --- OPTIMIZATION: Channels Last Memory Format ---
        self.model = model.to(self.device, memory_format=torch.channels_last)
        
        # --- OPTIMIZATION: Model Compilation (PyTorch 2.0+) ---
        try:
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile() for speed.")
        except Exception as e:
            self.logger.info(f"Skipping torch.compile: {e}")

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        
        self.train_loader, self.val_loader = get_dataloaders('full')
        
        # --- OPTIMIZER CHANGE ---
        # Fixed logic: CCT/ViT should almost always use AdamW.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # --- SCHEDULER & WARMUP ---
        # OneCycleLR includes a Warmup phase (pct_start=0.3 means 30% warmup).
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                epochs=num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,          # 30% Warmup phase (Critical for CCT)
                div_factor=25.0,        # Initial LR = max_lr / 25
                final_div_factor=1000.0 # Final LR = initial_LR / 1000
            )
            
        if use_label_smoothing:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info("CCT Trainer initialized.")
        
    def _setup_logging(self):
        """Setup logging to both console and file"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        log_file = log_dir / f"{self.timestamp}_Log_{self.model_name}.log"
        
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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def train_one_epoch(self, epoch):
        self.model.train()
        loss_total = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = images.to(self.device, memory_format=torch.channels_last)
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
            
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_total += loss.item() * labels.size(0)
            
        return loss_total / total, 100 * correct / total
    
    def validate(self, epoch):
        self.model.eval()
        loss_total, correct, total = 0.0, 0, 0

        do_cm = (self.cm_every > 0) and ((epoch+1) % self.cm_every == 0)
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(self.device, memory_format=torch.channels_last)
                labels = labels.to(self.device)
                
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_total += loss.item() * labels.size(0)

                if do_cm:
                    all_preds.append(predicted.detach().cpu())
                    all_labels.append(labels.detach().cpu())

        if do_cm and len(all_preds) > 0:
            labels_np = torch.cat(all_labels).numpy()
            preds_np  = torch.cat(all_preds).numpy()
            cm_path = create_cm(labels=labels_np,
                             preds=preds_np,
                             class_names=self.val_loader.dataset.classes,
                             epoch=epoch,
                             model_name=self.model_name,
                             timestamp=self.timestamp)
            self.logger.info(f"Confusion matrix saved at {cm_path}")
            
        return loss_total / total, 100 * correct / total
    
    def _print_epoch_summary(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy, current_lr):
        """Print formatted epoch summary"""
        self.logger.info("─" * 66)
        self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
        self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        self.logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        self.logger.info(f"  LR: {current_lr:.6f}")
        self.logger.info("─" * 66)
    
    def save_model(self, path: Path, model, optimizer, epoch: int, best_val_acc: float):
        model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        
        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "optim_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, path)    
        self.logger.info(f"💾 Saved best model (val_acc: {best_val_acc:.2f}%) to {path}")

    def train(self):
        self.logger.info(f"Beginning CCT training for {self.num_epochs} epochs.")
        self.logger.info("-" * 60)
        
        best_val_acc = -1.0
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            current_lr = self.scheduler.get_last_lr()[0] if self.use_scheduler else self.optimizer.param_groups[0]["lr"]
            self._print_epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.filepath, self.model, self.optimizer, epoch, best_val_acc)
            
            if early_stopping.check(val_acc):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                self.logger.info(f"No improvement in validation accuracy for {early_stopping.patience} epochs.")
                self.logger.info(f"Best validation accuracy: {early_stopping.best_value:.2f}%")
                self.logger.info(f"{'='*60}")
                break
                
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training complete.")