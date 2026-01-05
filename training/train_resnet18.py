"""
Implementierung eines Training-Loops für ResNet-18 (models/ResNet-18)

Beispielhafte Implementierung siehe Ende der Datei.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import ResNet18
from training.load_data import get_dataloaders
from training.early_stopping import EarlyStopping

class ResNetTrainer:
    
    def __init__(self, 
                 filepath: Path, # Filepath to save best model / checkpoints to
                 num_epochs: int = 3, 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 0.0001):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.filepath = filepath
        
        # Set Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using Device: {self.device}")
        
        # Model
        self.model = ResNet18()
        self.model = self.model.to(self.device)
        
        # Dataloaders
        self.train_loader, self.val_loader = get_dataloaders()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay)
        
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
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            predicted = outputs.argmax(dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss_total += loss.item() * labels.size(0)
            
        loss_avg = loss_total / total
        accuracy = 100 * correct / total
        return loss_avg, accuracy
    
    
    def validate(self):
        """Validate module on validation data"""
        self.model.eval()
        
        loss_total, correct, total = 0.0, 0, 0
        
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
            val_loss, val_accuracy = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
            
            # Store best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model(self.filepath, self.model, self.optimizer, epoch, best_val_acc)
                print(f"Saved Model to {self.filepath}")
            
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
        
        
if __name__ == "__main__":
    # Test Training
    # Save best model to checkpoint folder
    dir =  Path("models") / "ResNet18" / "checkpoints"
    dir.mkdir(parents=True, exist_ok=True)
    chk_path = dir / "best.pt"
    trainer = ResNetTrainer(chk_path)
    trainer.train()