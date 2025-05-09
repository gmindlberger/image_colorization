import itertools
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd

from trainer import Trainer
from dataloader import ImageColorizerDataset
from neural_nets import ImageColorizer


def run_grid_search(
    image_path: str,
    grid: dict,
    device: torch.device = None,
    val_fraction: float = 0.1,
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
        print(f"Testing size={img_size}, lr={lr}, bs={bs}, epochs={epochs}")

        # Dataset und Split
        ds = ImageColorizerDataset(image_path, *img_size)
        n_val = int(len(ds) * val_fraction)
        train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=bs)

        # Modell & Training
        model = ImageColorizer(1, 3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, train_loader, val_loader, torch.nn.MSELoss(), optimizer, device)
        trainer.fit(epochs)

        val_loss = trainer.history['val_loss'][-1]
        results.append({
            'img_size': f"{img_size[0]}x{img_size[1]}",
            'lr': lr,
            'bs': bs,
            'epochs': epochs,
            'val_loss': val_loss
        })

        if val_loss < best_val:
            best_val = val_loss
            best_cfg = (img_size, lr, bs, epochs)
    df = pd.DataFrame(results)
    print(f"Grid Search abgeschlossen. Beste Konfiguration: {best_cfg}, loss={best_val:.4f}")
    return df


def plot_results(df: pd.DataFrame) -> None:
    """
    Plottet die minimalen Validation-Loss-Werte pro Bildgröße.
    """
    grouped = df.groupby('img_size')['val_loss'].min().reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(grouped['img_size'], grouped['val_loss'])
    plt.xlabel('Image Size')
    plt.ylabel('Min Validation Loss')
    plt.title('Best Validation Loss by Image Size')
    plt.tight_layout()
    plt.show()
