from models import ResNet18
from models import ResNet34
from models import ResNet18_SE
from training.ResNetTrainer import ResNetTrainer

if __name__ == '__main__':
    model = ResNet34() 
    trainer = ResNetTrainer(model)
    trainer.train()

