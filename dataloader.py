from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageColorizerDataset(Dataset):
    def __init__(self, img_dir, img_height=256, img_width=256):
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg'))
        ])

        # define Transformations
        transform_gray = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        transform_color = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])

        self.transform_gray = transform_gray
        self.transform_color = transform_color

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # color target
        target = img.copy()
        if self.transform_color:
            target = self.transform_color(target)
        # grayscale input
        gray = img.convert('L')
        if self.transform_gray:
            gray = self.transform_gray(gray)
        return gray, target