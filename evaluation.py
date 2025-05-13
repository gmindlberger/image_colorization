import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000

def evaluate_psnr(model, loader, device, n=20):
    """
        Berechnet den durchschnittlichen PSNR-Wert für n Bilder.
        Der PSNR-Wert (Peak Signal-to-Noise Ratio) ist eine Metrik aus der Signalverarbeitung und misst, wie stark ein
        rekonstruiertes Bild vom Original abweicht – bezogen auf den maximal möglichen Pixelwert.
        -> PSNR ist nicht wahrnehmungsnah: zwei Bilder mit sehr unterschiedlichem Aussehen können den gleichen PSNR haben, wenn die Pixelabweichungen gleich sind.
        -> Berechnung mithilfe des MeanSquaredErrors
    """
    # Setze das Modell in den Evaluierungsmodus, statt in den Trainingsmodus
    # (Deaktiviert dropout (schutz vor overfitting/deaktiviert neuronen)
    # fixiert Batchnorm (berechnet während des Trainings Statistiken aus dem aktuellen Batch (Mittelwert & Standardabweichung))
    model.to(device).eval()
    # Initialisiere Variablen für die Berechnung des Durchschnitts
    total, count = 0, 0
    # iterieren über den dataloader -> testloader wurde hierfür übergeben (Test-Datensätze)
    for L, AB in loader:
        # Am besten mit CUDA arbeiten
        L = L.to(device)
        # no_grad -> Keine Gradientenberechnung spart Performance und ist nicht nötig (da kein Training, sondern Vorhersage)
        # Vorhersage des Modells erstellen mit pred, anhand der L-Werte
        # Anschließend konvertierung des Tensors in Numpy. Hierfür ist die CPU wieder nötig -> Darum auf die CPU.
        with torch.no_grad(): pred = model(L).cpu().numpy()
        # Die AB-werte in Numpy Array umwandeln
        true = AB.numpy()
        # L-Kanal in uint8 umwandeln (0-255) und auf die CPU zurückschieben für Numpy-Array
        L = (L.cpu().numpy() * 255).astype(np.uint8)
        # Da die Bilder in batches verarbeitet werden, muss die Bedingung eingebaut werden.
        # Pro For Schleife, also maximal 8 (für 1 Batch), aber auch nur max 20 weil n=20
        for i in range(min(len(pred), n - count)):
            # Formatieren des Bildes das dran ist in RGB
            pred_rgb = lab2rgb(L[i,0], pred[i])
            true_rgb = lab2rgb(L[i,0], true[i])
            # Berechnung des PSNR-Wertes anhand des echten Wertes und dem Predicteten Wertes.
            total += peak_signal_noise_ratio(true_rgb, pred_rgb)
            count += 1
        if count >= n: break
    return total / count

def evaluate_ssim(model, loader, device, n=20):
    """
        Berechnet den durchschnittlichen SSIM-Wert für n Bilder.

        Der SSIM-Wert (Structural Similarity Index) misst, wie ähnlich zwei Bilder in ihrer Struktur sind.
        Relevante Indikatoren hierfür sind Helligkeit (Luminanz), Kontrast und Struktur (z.B. Kanten und Muster).

    """
    #Afbau der Funktion analog oben -> Unterschied ist die aufgerufene Funktion aus Scikit Learn Image
    model.to(device).eval()
    total, count = 0, 0
    for L, AB in loader:
        L = L.to(device)
        with torch.no_grad(): pred = model(L).cpu().numpy()
        true = AB.numpy()
        L = (L.cpu().numpy() * 255).astype(np.uint8)
        for i in range(min(len(pred), n - count)):
            pred_rgb = lab2rgb(L[i,0], pred[i])
            true_rgb = lab2rgb(L[i,0], true[i])
            total += structural_similarity(true_rgb, pred_rgb, channel_axis=-1)
            count += 1
        if count >= n: break
    return total / count

def lab2rgb(L, ab):
    """
        Hilfsfunktion: Wandelt L-Kanal + AB-Kanäle (LAB) in RGB-Bild um.

        Args:
            L: Graustufenbild (2D, uint8)
            ab: Farbinformationen A+B (2xHxW, float in [0,1])

        Returns:
            RGB-Bild als numpy-Array
    """
    # zurück skalieren der AB Werte
    ab = (ab * 255).astype(np.uint8)
    # Die einzelnen Kanäle werden gemerged, um ein LAB-Bild zu erstellen
    lab = cv2.merge([L, ab[0], ab[1]])
    # Rückgabe der RGB-Darstellung
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def evaluate_ciede2000(model, loader, device, n=20):
    """
    Berechnet den durchschnittlichen CIEDE2000-Farbunterschied für n Bilder.

    CIEDE2000 ist eine wahrnehmungsnahe Metrik für Farbunterschiede im LAB-Farbraum.
    Ein Wert von 0 bedeutet perfekte Übereinstimmung. Werte < 2.3 gelten als kaum wahrnehmbar.

    CIEDE2000 berechnet den Farbunterschied zwischen zwei LAB-Farben, indem es Helligkeit,
    Farbton und Sättigung einzeln vergleicht und diese Unterschiede mit Wahrnehmungsfaktoren gewichtet.
    Dadurch entsteht ein einziger Wert (ΔE₀₀), der beschreibt, wie unterschiedlich zwei Farben für das menschliche Auge wirken.

    """
    model.to(device).eval()
    total, count = 0, 0
    for L, AB in loader:
        L = L.to(device)
        with torch.no_grad(): pred = model(L).cpu().numpy()
        true = AB.numpy()
        L = (L.cpu().numpy() * 255).astype(np.uint8)
        for i in range(min(len(pred), n - count)):
            # Holt die Graustufenwerte L eines Bildes raus)
            L_img = L[i, 0]
            # Erstellt das LAB-Bild anhand der vorhandenen Werte. diese müssen dieses mal als float32 Formatiert sein für die Berechnung
            # axis 0 2 weil wir die Farben auch übergeben.
            # Die Werte werden gestapelt, damit man die Struktur Höhe x Breite x 3 Farbkanäle erhält
            pred_lab = np.stack([L_img, pred[i][0] * 255, pred[i][1] * 255], axis=2).astype(np.float32)
            true_lab = np.stack([L_img, true[i][0] * 255, true[i][1] * 255], axis=2).astype(np.float32)
            # Übergabe in die Scikit-Learn Image Funktion
            delta = deltaE_ciede2000(pred_lab, true_lab)
            total += delta.mean()
            count += 1
        if count >= n: break
    return total / count
