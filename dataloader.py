import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageColorizerDataset(Dataset):
    def __init__(self, img_dir, img_height=256, img_width=256):
        # alle .jpg-Dateien sammeln
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith('.jpg')
        ])

        # L bleibt in [0,1]
        self.transform_L = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),   # skaliert 0–255 → 0.0–1.0
        ])

        # AB wird durch OpenCV schon auf 0–255 verschoben
        # → ToTensor() skaliert also automatisch (AB+128)/255 → 0.0–1.0
        self.transform_AB = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),   # skaliert 0–255 → 0.0–1.0
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Bild laden und zu RGB konvertieren
        path    = self.paths[idx]
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # RGB → LAB; OpenCV-LAB ist 0–255, A/B sind automatisch um +128 verschoben
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(img_lab)

        # PIL-Images erzeugen
        L_img  = Image.fromarray(L).convert('L')                              # 1-Channel
        AB_img = Image.fromarray(np.stack([A, B], axis=2).astype(np.uint8))  # 2-Channel

        # Transforms anwenden
        L_tensor  = self.transform_L(L_img)   # 1×H×W in [0,1]
        AB_tensor = self.transform_AB(AB_img) # 2×H×W in [0,1]

        return L_tensor, AB_tensor
