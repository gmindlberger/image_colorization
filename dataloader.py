import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageColorizerDataset(Dataset):
    def __init__(self, img_dir, img_height=256, img_width=256):
        # alle .jpg-Dateien sammeln
        # sortierte Liste aller Bildpfade
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith('.jpg')
        ])

        # L bleibt in [0,1] // transformierung für L-Kanal
        self.transform_L = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),   # skaliert 0–255 → 0.0–1.0
        ])

        # AB bleibt in [0,1] // transformierung für AB-Kanäle
        # ToTensor() skaliert also automatisch (AB+128)/255 → 0.0–1.0 + 128 automatisch durhc OpenCV durchgeführt.
        # Um 128 verschieben, da die Range von -128 bis 127 geht
        # A grün -> Magenta
        # B blau -> gelb
        self.transform_AB = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Bild laden und zu RGB konvertieren
        path    = self.paths[idx]
        # OpenCV liest in BGR
        img_bgr = cv2.imread(path)
        # OpenCV muss die Farben in RGB konvertieren
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # RGB -> LAB; OpenCV-LAB ist 0–255, A/B sind automatisch um -128 verschoben
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        # in die LAB-Kanäle splitten
        L, A, B = cv2.split(img_lab)

        # PIL-Images erzeugen
        #Erzeugt das Graustufenbild L, stellt sicher, dass es wirklich ein 1-Kanal-Bild ist
        L_img  = Image.fromarray(L).convert('L')
        # Stapel die A und B zu einem Bild mit 2 Kanälen -> astype(uint8), notwendiges Format (split hat ausgaben wir int32 und float64 -> Schlecht für PIL)
        AB_img = Image.fromarray(np.stack([A, B], axis=2).astype(np.uint8))  # 2-Channel

        # Transforms anwenden
        # Siehe oben -> Resize, skalieren, ToTensor
        L_tensor  = self.transform_L(L_img)   # 1×H×W in [0,1]
        AB_tensor = self.transform_AB(AB_img) # 2×H×W in [0,1]

        return L_tensor, AB_tensor
