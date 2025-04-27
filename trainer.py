import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 criterion,
                 optimizer,
                 device: torch.device):
        self.device = device
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer

        # Verlaufs‐Logs
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_epoch(self):
        self.model.train()
        running = 0.0
        for gray, target in tqdm(self.train_loader, desc='  Train', leave=False):
            gray, target = gray.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(gray)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * gray.size(0)
        return running / len(self.train_loader.dataset)

    def validate_epoch(self):
        self.model.eval()
        running = 0.0
        with torch.no_grad():
            for gray, target in tqdm(self.val_loader, desc='  Val  ', leave=False):
                gray, target = gray.to(self.device), target.to(self.device)
                pred = self.model(gray)
                loss = self.criterion(pred, target)
                running += loss.item() * gray.size(0)
        return running / len(self.val_loader.dataset)

    def fit(self, num_epochs: int):
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_epoch()
            val_loss   = self.validate_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            print(f'Epoch {epoch}/{num_epochs} — '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    def plot_history(self):
        epochs = range(1, len(self.history['train_loss'])+1)
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.history['train_loss'], '-o', label='Train')
        plt.plot(epochs, self.history['val_loss'],   '-s', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
