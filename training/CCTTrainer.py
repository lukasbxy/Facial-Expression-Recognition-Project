"""
Implementierung eines Training-Loops für CCT (models/CCT)
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
        
        # --- OPTIMIZATION: CUDA Benchmark ---
        torch.backends.cudnn.benchmark = True
        
        # Automatically construct filepath
        model_name = model.__class__.__name__
        self.filepath = Path("models") / model_name / "checkpoints" / "best.pt"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Set Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using Device: {self.device}")
        
        # --- OPTIMIZATION: Channels Last Memory Format ---
        self.model = model.to(self.device, memory_format=torch.channels_last)
        
        # --- OPTIMIZATION: Model Compilation (PyTorch 2.0+) ---
        try:
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile() for speed.")
        except Exception as e:
            print(f"Skipping torch.compile: {e}")

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
        
        print("CCT Trainer initialized.")
        
    def train_one_epoch(self):
        self.model.train()
        loss_total = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
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
            for images, labels in tqdm(self.val_loader, desc="Validating"):
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
                             epoch=epoch)
            print(f"Confusion matrix saved at {cm_path}")
            
        return loss_total / total, 100 * correct / total
    
    def save_model(self, path: Path, model, optimizer, epoch: int, best_val_acc: float):
        model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        
        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "optim_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, path)    
        print(f"Saved Model to {path}")

    def train(self):
        print(f"Beginning CCT training for {self.num_epochs} epochs.")
        print("-" * 60)
        
        best_val_acc = -1.0
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate(epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            if self.use_scheduler:
                print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}") 
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.filepath, self.model, self.optimizer, epoch, best_val_acc)
            
            if early_stopping.check(val_acc):
                print(f"\nEarly stopping triggered. Best Acc: {early_stopping.best_value:.2f}%")
                break
                
        print("\n" + "=" * 60)
        print("Training complete.")