import torch
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader = None,
                 criterion=None,
                 optimizer=None,
                 device: torch.device = torch.device('cpu'),
                 validate: bool = False):
        self.device       = device
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.validate     = validate

        # Verlaufs‐Logs
        self.history = {
            'train_loss': []
        }
        if self.validate:
            self.history['val_loss'] = []

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
        best_loss = float('inf')
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            if self.validate:
                val_loss = self.validate_epoch()
                self.history['val_loss'].append(val_loss)
                current_primary = val_loss
                msg = f"Epoch {epoch}/{num_epochs} — Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            else:
                current_primary = train_loss
                msg = f"Epoch {epoch}/{num_epochs} — Train: {train_loss:.4f}"

            # Best-Model speichern (nach Val oder nach Train, je nach Mode)
            if current_primary < best_loss:
                best_loss = current_primary
                torch.save(self.model.state_dict(), "best_model_save.pth")

            print(msg)

    def plot_history(self):
        epochs = range(1, len(self.history['train_loss'])+1)
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.history['train_loss'], '-o', label='Train')
        if self.validate:
            plt.plot(epochs, self.history['val_loss'], '-s', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_examples(self, loader=None, n_display=3):
        if loader is None:
            loader = self.val_loader if self.validate and self.val_loader is not None else self.train_loader

        self.model.eval()
        with torch.no_grad():
            L_batch, AB_batch = next(iter(loader))
            L_batch = L_batch.to(self.device)
            pred_AB = self.model(L_batch).cpu()

        L_np       = (L_batch.cpu().squeeze(1).numpy() * 255).astype(np.uint8)
        pred_AB_np = (pred_AB.numpy() * 255).astype(np.uint8)
        true_AB_np = (AB_batch.numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(n_display, 3, figsize=(12, 4 * n_display))
        for i in range(n_display):
            # 1) Input L
            axes[i,0].imshow(L_np[i], cmap='gray')
            axes[i,0].axis('off')
            axes[i,0].set_title('Input L')

            # 2) Predicted RGB
            lab_pred = cv2.merge([L_np[i], pred_AB_np[i,0], pred_AB_np[i,1]])
            rgb_pred = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2RGB)
            axes[i,1].imshow(rgb_pred)
            axes[i,1].axis('off')
            axes[i,1].set_title('Predicted RGB')

            # 3) Ground Truth
            lab_true = cv2.merge([L_np[i], true_AB_np[i,0], true_AB_np[i,1]])
            rgb_true = cv2.cvtColor(lab_true, cv2.COLOR_LAB2RGB)
            axes[i,2].imshow(rgb_true)
            axes[i,2].axis('off')
            axes[i,2].set_title('Ground Truth')

        plt.tight_layout()
        plt.show()
