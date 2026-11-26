import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# wczytanie danych
def get_spectrograms_dir(dataset):
    env_path = os.getenv("SPECTROGRAMS_DIR")
    if env_path:
        path = Path(env_path) / str(dataset)
        if path.exists():
            return path
        raise FileNotFoundError(f"Brak danych w SPECTROGRAMS_DIR: {path}")

    candidates = [
        Path(r"C:\spectrograms"),
        Path(__file__).resolve().parents[1] / "spectrograms",
    ]
    for root in candidates:
        path = root / str(dataset)
        if path.exists():
            return path
        
    raise FileNotFoundError(
        f"Nie znaleziono datasetu '{dataset}' w żadnej z lokalizacji: {candidates}"
    )

#PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(spectrograms):
    labels = spectrograms / "etykiety.csv"
    data = pd.read_csv(labels)
    return data

INSTRUMENTS = {
    "Acoustic Guitar (steel)": "Gitara akustyczna",
    "Alto Sax": "Saksofon altowy",
    "Bright Acoustic Piano": "Fortepian",
    "Drums": "Perkusja",
    "Electric Bass (finger)": "Gitara basowa",
    "Flute": "Flet",
    "String Ensemble 1": "Sekcja smyczkowa"
}

class InstrumentDataset(Dataset):
    """
    Klasa tworzy dataset, w której w każdej próbce występuje mel-spektogram oraz
    wektor binarnych etykiet określających obecność instrumentu.
    """

    def __init__(self, data, root, zbior):
        self.data = (
            data[data["dataset"] == zbior]
            .reset_index(drop=True)
        )
        self.root = root
        self.zbior = zbior

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        nazwa_pliku = Path(row["audio_file"]).with_suffix(".npy").name
        sciezka = self.root / self.zbior / nazwa_pliku

        spec = np.load(sciezka)
        spec = torch.from_numpy(spec).float().unsqueeze(0)

        label = row[list(INSTRUMENTS.keys())].to_numpy(dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return spec, label

# Utworzenie zbiorów danych
def create_datasets(dataset, spectrograms_root=None):
    
    if spectrograms_root is not None:
        spectrograms = Path(spectrograms_root) / str(dataset)
    else:
        spectrograms = get_spectrograms_dir(dataset)

    print(f"Wczytuję spektrogramy z: {spectrograms}")
    data = load_data(spectrograms)

    train_dataset = InstrumentDataset(data, spectrograms, "train")
    val_dataset = InstrumentDataset(data, spectrograms, "validation")
    test_dataset = InstrumentDataset(data, spectrograms, "test")

    return train_dataset, val_dataset, test_dataset

# Utworzenie DataLoaderów
def create_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, seed=6709):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader