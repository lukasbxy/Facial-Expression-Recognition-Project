"""
Implementierung eines Training-Loops für ResNet-18 (models/ResNet-18)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ResNet18.ResNet18 import ResNet18
from training.load_data import get_dataloaders

class ResNetTrainer:
    
    def __init__(self, num_epochs: int = 10, learning_rate: float = 0.001, weight_decay: float = 0.0001):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            
        loss_avg = loss_total / len(self.train_loader)
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
            
        loss_avg = loss_total / len(self.val_loader)
        accuracy = 100 * correct / total
        return loss_avg, accuracy
                

    def train(self):
        """Main training loop"""
        print(f"Beginning training for {self.num_epochs} epochs.")
        print(60*"-")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        print("\n" + "=" * 60)
        print("Training complete.")
        
if __name__ == "__main__":
    # Test Training
    trainer = ResNetTrainer()
    trainer.train()