import torch
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Trainer:
    # Initilisierung der Klasse und übernahme der Parameter
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader = None,
                 criterion=None,
                 optimizer=None,
                 device: torch.device = torch.device('cpu'),
                 validate: bool = False):
        # Device wählen und Model reinhauen
        self.device = device
        self.model = model.to(device)  # Model auf GPU/CPU packen
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.validate = validate

        # History für Loss speichern
        self.history = {'train_loss': []}
        # Wenn Validierung == True ist, dann auch den val_loss
        if self.validate:
            self.history['val_loss'] = []  # Val-Loss nur bei Bedarf

    def train_epoch(self):
        # Das Modell soll trainiert werden.
        self.model.train()  # Trainingsmodus aktivieren
        running = 0.0 # Variable initialisieren
        #iterieren über die traingsdaten
        for gray, target in tqdm(self.train_loader, desc='  Train', leave=False):
            # Verschieben der Daten auf das gewählte Device
            gray, target = gray.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()  # Gradienten nullen (damit dieses sich nicht über die Batches aufsummieren)
            pred = self.model(gray)  # Vorhersage aufgrund der Eingabedaten
            loss = self.criterion(pred, target)  # Loss berechnen
            loss.backward()  # Backprop; berechnet die Gradienten der Gewichte basierend auf dem Loss
            self.optimizer.step()  # Update-Step der Gewichte mit Hilfe der Gradienten
            running += loss.item() * gray.size(0) # konvertiert den Tensor in eine Python Zahl - Produkt gibt gesamten Loss des Batches
        return running / len(self.train_loader.dataset)  # Avg-Loss als Rückgabe

    def validate_epoch(self):
        # Validierung des Modells (ohne Training), mit den Validierungsdaten
        # Ist wichtig, damit man mögliches overfitting aus dem Training erkennt. Es "validiert" nochmal das vorhandene Modell.
        self.model.eval()  # Evalmodus einschalten --> Genaue erklärung in evaluation.
        running = 0.0
        with torch.no_grad(): # Gradientenberechnung deaktivieren
            for gray, target in tqdm(self.val_loader, desc='  Val  ', leave=False):
                gray, target = gray.to(self.device), target.to(self.device)
                pred = self.model(gray) # Vorhersage berechnen
                loss = self.criterion(pred, target) # loss berechnen
                running += loss.item() * gray.size(0) # loss aufsummieren
        return running / len(self.val_loader.dataset) # Avg-Loss als Rückgabe

    # Fit ruft die Trainings und Validierungs Epochen auf -> So wird alles kombiniert und gestartet.
    def fit(self, num_epochs: int):
        best_loss = float('inf') # Variable initialisieren
        # Schleife über die Anzahl an Epochen
        for epoch in range(1, num_epochs + 1):
            # Trainieren des Modells
            train_loss = self.train_epoch()
            # Train Loss speichern
            self.history['train_loss'].append(train_loss)

            # Durchführen einer Validierungsepoche + Abspeichern des Modells, wenn der Loss geringer als der beste bisherige ist
            if self.validate:
                val_loss = self.validate_epoch()
                self.history['val_loss'].append(val_loss)
                current_primary = val_loss
                msg = f"Epoch {epoch}/{num_epochs} — Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            else:
                current_primary = train_loss
                msg = f"Epoch {epoch}/{num_epochs} — Train: {train_loss:.4f}"
            # Bestes Model sichern
            if current_primary < best_loss:
                best_loss = current_primary
                torch.save(self.model.state_dict(), "best_model_save.pth")  # Save

            print(msg)

    # Funktion zur Ausgabe der Loss-Werte über die Trainingsepochen (jeweils Trainings- und Validierungsloss (falls gewollt))
    def plot_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_loss'], '-o', label='Train')  # Trainings-Loss
        if self.validate:
            plt.plot(epochs, self.history['val_loss'], '-s', label='Val')  # Val-Loss
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()  # Chart zeigen

    def show_examples(self, loader=None, n_display=3):
        # Default: Validation, sonst Training wenn kein validation Datensatz vorhanden ist.
        if loader is None:
            loader = self.val_loader if self.validate and self.val_loader else self.train_loader

        # Das modell in den Evaluationsmodus versetzen (siehe evaluation für genauere erklärung)
        self.model.eval()
        with torch.no_grad(): # deaktivieren des Gradientenberechnung
            L_batch, AB_batch = next(iter(loader))  # Eine Batch ziehen, loader ist iterierbar, und next nimmt den nächsten raus
            L_batch = L_batch.to(self.device)
            pred_AB = self.model(L_batch).cpu()  # Vorhersage zurück auf CPU

        # Konvertieren zu NumPy & Skalieren
        L_np = (L_batch.cpu().squeeze(1).numpy() * 255).astype(np.uint8)
        pred_AB_np = (pred_AB.numpy() * 255).astype(np.uint8)
        true_AB_np = (AB_batch.numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(n_display, 3, figsize=(12, 4 * n_display))
        for i in range(n_display):
            # Graustufen-Input
            axes[i, 0].imshow(L_np[i], cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Input L')

            # Predicted RGB aus LAB
            lab_pred = cv2.merge([L_np[i], pred_AB_np[i, 0], pred_AB_np[i, 1]])
            rgb_pred = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2RGB)
            axes[i, 1].imshow(rgb_pred)
            axes[i, 1].axis('off')
            axes[i, 1].set_title('Predicted RGB')

            # Ground Truth
            lab_true = cv2.merge([L_np[i], true_AB_np[i, 0], true_AB_np[i, 1]])
            rgb_true = cv2.cvtColor(lab_true, cv2.COLOR_LAB2RGB)
            axes[i, 2].imshow(rgb_true)
            axes[i, 2].axis('off')
            axes[i, 2].set_title('Ground Truth')

        plt.tight_layout()
        plt.show()  # Beispiele anzeigen
