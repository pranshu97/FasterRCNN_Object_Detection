from src.dataloader import train_loader, val_loader
from src.model import model, trainer

if __name__=='__main__':
    trainer.fit(model, train_loader, val_loader)
