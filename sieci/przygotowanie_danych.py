
# biblioteki
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# wczytanie danych
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECTROGRAMS = (
    PROJECT_ROOT / "spectrograms" / "mels80_frame1024_step512_windowhann_sr44100_4s"
)
LABELS = SPECTROGRAMS / "etykiety.csv"


def load_data():
    data = pd.read_csv(LABELS)
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
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        label = row[list(INSTRUMENTS.keys())].to_numpy(dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return spec, label


# Utworzenie zbiorów danych
def create_datasets():
    data = load_data()

    train_dataset = InstrumentDataset(data, SPECTROGRAMS, "train")
    val_dataset = InstrumentDataset(data, SPECTROGRAMS, "validation")
    test_dataset = InstrumentDataset(data, SPECTROGRAMS, "test")

    return train_dataset, val_dataset, test_dataset


# Utworzenie DataLoaderów
def create_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=8,
    seed=6709
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
