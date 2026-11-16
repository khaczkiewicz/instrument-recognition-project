from pathlib import Path
import pandas as pd

import spectrograms_preparation as prep

BASE_DIR = Path(__file__).resolve().parent

CONFIGURATIONS_CSV = BASE_DIR.parent / "spectrograms" / "konfiguracje.csv"
DATA_BASE_DIR = BASE_DIR.parent / "data" / "slakh"
OUTPUT_BASE_DIR = BASE_DIR.parent / "spectrograms"

def main():

    configurations = pd.read_csv(CONFIGURATIONS_CSV, sep=";")

    for _, row in configurations.iterrows():

        configuration_id = int(row["configuration"])
        data_id = str(row["data"])

        dataset_dir = DATA_BASE_DIR / data_id
        output_dir = OUTPUT_BASE_DIR / f"{data_id}_{configuration_id}"



        prep.generate_spectrograms_from_dataset(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            sample_rate=44100,
            expected_duration=4.0,
            overwrite=False,
            n_mel_banks=int(row["n_mel_banks"]),
            frame_length=int(row["frame_length"]),
            step_length=int(row["step_length"]),
        )


if __name__ == "__main__":
    main()
