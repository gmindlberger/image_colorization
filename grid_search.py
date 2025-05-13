import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
from trainer import Trainer
from dataloader import ImageColorizerDataset
from neural_nets import ImageColorizerLAB

# Dieser Grid Search code ist im Grunde nur eine Kopie des OriginalCodes in AutoColorization.ipynb
# Allerdings ist es bewusst vollständig ausgelagert - Dementsprechend auch hier keine weiteren Details zum Code per se.

def run_grid_search(

    image_path: str,
    grid: dict,
    device: torch.device = None,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> pd.DataFrame:
    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    results = []
    best_val = float('inf')
    best_cfg = None

    for img_size, lr, bs, epochs in itertools.product(
            grid['img_size'], grid['lr'], grid['bs'], grid['epochs']
    ):
        print(f"\nTesting size={img_size}, lr={lr}, bs={bs}, epochs={epochs}")
        dataset = ImageColorizerDataset(image_path, *img_size)
        n_total = len(dataset)
        n_val = int(n_total * val_fraction)
        n_test = int(n_total * test_fraction)
        n_train = n_total - n_val - n_test

        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)
        model = ImageColorizerLAB(1, 2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device, validate=True)
        trainer.fit(epochs)

        train_loss = trainer.history['train_loss'][-1]
        val_loss = trainer.history['val_loss'][-1]
        results.append({
            'img_size': f"{img_size[0]}x{img_size[1]}",
            'lr': lr,
            'bs': bs,
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    df = pd.DataFrame(results)
    return df


def plot_results(df: pd.DataFrame) -> None:
    """
    Plottet die minimalen Validation-Loss-Werte pro Bildgröße.
    """
    grouped = df.groupby('img_size')[['train_loss', 'val_loss']].min().reset_index()
    plt.figure(figsize=(8, 5))
    x = grouped['img_size']
    plt.bar(x, grouped['train_loss'], label='Train Loss', alpha=0.6)
    plt.bar(x, grouped['val_loss'], label='Val Loss', alpha=0.6)
    plt.xlabel('Image Size')
    plt.ylabel('Loss')
    plt.title('Best Train & Val Loss by Image Size')
    plt.legend()
    plt.tight_layout()
    plt.show()
