from pathlib import Path
import pandas as pd
import shutil

import slakh_preparation as prep

BASE_DIR = Path(__file__).resolve().parent

CONFIGURATIONS_CSV = BASE_DIR.parent / "data" / "slakh" / "konfiguracje.csv"
OUTPUT_BASE_DIR = CONFIGURATIONS_CSV.parent


def main():

    configurations = pd.read_csv(CONFIGURATIONS_CSV, sep=";")

    for _, row in configurations.iterrows():

        configuration_id = str(row["configuration"])
        output_dir = OUTPUT_BASE_DIR / configuration_id

        if output_dir.exists():
            continue

        audio_config = {
            "SELECTED_INSTRUMENTS": {
                instrument.strip()
                for instrument in row["SELECTED_INSTRUMENTS"].split("|")
            },
            "SEGMENT_LENGTH": float(row["SEGMENT_LENGTH"]),
            "ENERGY_THRESHOLD": float(row["ENERGY_THRESHOLD"]),
            "NOISE_LEVEL": float(row["NOISE_LEVEL"]),
            "TARGET_PER_CATEGORY": int(row["TARGET_PER_CATEGORY"]),
            "TRAIN_RATIO": float(row["TRAIN_RATIO"]),
            "VALIDATION_RATIO": float(row["VALIDATION_RATIO"]),
            "TEST_RATIO": float(row["TEST_RATIO"]),
        }

        prep.generate_audio(**audio_config)

        output_dir.mkdir(parents=True)

        for dataset_name in ["train", "validation", "test"]:
            source_dir = prep.DATASET_DIR / dataset_name
            destination_dir = output_dir / dataset_name

            shutil.copytree(source_dir, destination_dir)

        labels_file = prep.DATASET_DIR / "etykiety.csv"

        if labels_file.exists():
            shutil.copy2(labels_file, output_dir / "etykiety.csv")

if __name__ == "__main__":
    main()
