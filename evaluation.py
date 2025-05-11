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
    """
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
    ab = (ab * 255).astype(np.uint8)
    lab = cv2.merge([L, ab[0], ab[1]])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def evaluate_ciede2000(model, loader, device, n=20):
    """
    Berechnet den durchschnittlichen CIEDE2000-Farbunterschied für n Bilder.

    CIEDE2000 ist eine wahrnehmungsnahe Metrik für Farbunterschiede im LAB-Farbraum.
    Ein Wert von 0 bedeutet perfekte Übereinstimmung. Werte < 2.3 gelten als kaum wahrnehmbar.

    """
    model.to(device).eval()
    total, count = 0, 0
    for L, AB in loader:
        L = L.to(device)
        with torch.no_grad():
            pred = model(L).cpu().numpy()
        true = AB.numpy()
        L = (L.cpu().numpy() * 255).astype(np.uint8)

        for i in range(min(len(pred), n - count)):
            L_img = L[i, 0]
            pred_lab = np.stack([L_img, pred[i][0] * 255, pred[i][1] * 255], axis=2).astype(np.float32)
            true_lab = np.stack([L_img, true[i][0] * 255, true[i][1] * 255], axis=2).astype(np.float32)

            delta = deltaE_ciede2000(pred_lab, true_lab)
            total += delta.mean()
            count += 1
        if count >= n: break
    return total / count
