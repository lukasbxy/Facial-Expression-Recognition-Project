"""
Implementierung eines Training-Loops für ResNet-18 (models/ResNet-18)

Beispielhafte Implementierung siehe Ende der Datei.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from  training.create_cm import create_cm

from training.load_data import get_dataloaders
from training.early_stopping import EarlyStopping

class ResNetTrainer:
    
    def __init__(self, 
                 model,  # Neural network model to train
                 num_epochs: int = 3, 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001,
                 cm_every: int = 5,
                 use_adamw: bool = False,
                 use_scheduler: bool = False,
                 use_label_smoothing: bool = False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cm_every = cm_every
        self.use_scheduler = use_scheduler
        
        # Automatically construct filepath based on model class name
        model_name = model.__class__.__name__
        self.filepath = Path("models") / model_name / "checkpoints" / "best.pt"
        
        # Create checkpoint directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Set Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using Device: {self.device}")
        
        # Model
        self.model = model
        self.model = self.model.to(self.device)
        
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device = self.device.type, enabled = self.use_amp) # type: ignore
        
        # Dataloaders
        self.train_loader, self.val_loader = get_dataloaders()
        
        # Optimizer
        # Adam or AdamW
        if use_adamw:    
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr = self.learning_rate,
                weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.AdamW(
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
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print("Trainer initialized.")
        
    def train_one_epoch(self):
        """
        Train model for one epoch.
        """
        
        self.model.train()
        
        loss_total = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(self.train_loader, desc = "Training"):
            # Load images to GPU/CPU
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
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
        self.model.eval()
        
        loss_total, correct, total = 0.0, 0, 0

        do_cm = (self.cm_every > 0) and ((epoch+1) % self.cm_every == 0) or epoch == 0 
        if do_cm:
            all_preds = [] 
            all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc = "Validating"):
                # Load images to GPU/CPU
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                predicted = outputs.argmax(dim = 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_total += loss.item() * labels.size(0)

                # Append predictions and labels 
                if do_cm:
                    all_preds.append(predicted.detach().cpu())
                    all_labels.append(labels.detach().cpu())

        # Create Confusion Matrix if wanted
        if do_cm:
            labels_np = torch.cat(all_labels).numpy()
            preds_np  = torch.cat(all_preds).numpy()
            cm_path = create_cm(labels=labels_np,
                             preds=preds_np,
                             class_names = self.val_loader.dataset.classes,
                             epoch = epoch)
            print(f"Confusion matrix saved at {cm_path}")
            
        loss_avg = loss_total / total
        accuracy = 100 * correct / total
        return loss_avg, accuracy
    
    
    def save_model(self, path: Path, model, optimizer, epoch: int, best_val_acc: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            path,
        )    
        print(f"Saved Model to {path}")
        

    def train(self):
        """Main training loop"""
        print(f"Beginning training for {self.num_epochs} epochs.")
        print(60*"-")
        
        # Store best accuracy during run to determine when to save model.
        best_val_acc = -1.0
        
        # Initialize Early Stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate(epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
            
            if self.use_scheduler:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"LR: {current_lr:.6f}") 
            
            # Store best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model(self.filepath, self.model, self.optimizer, epoch, best_val_acc)
            
            # Check Early Stopping
            if early_stopping.check(val_accuracy):
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs.")
                print(f"No improvement in validation accuracy for {early_stopping.patience} epochs.")
                print(f"Best validation accuracy: {early_stopping.best_value:.2f}%")
                print(f"{'='*60}")
                break
                
        print("\n" + "=" * 60)
        print("Training complete.")