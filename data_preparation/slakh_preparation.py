import tarfile
import yaml
from pathlib import PurePosixPath, Path
from collections import defaultdict
import shutil
import re
import csv
import numpy as np
import soundfile as sf
import random
from itertools import combinations

random.seed(2137)

# Ustawienia

SELECTED_INSTRUMENTS = set()

SEGMENT_LENGTH = 4.0
ENERGY_THRESHOLD = 0.05
NOISE_LEVEL = 0.0

TARGET_PER_CATEGORY = 100

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

instruments = sorted(SELECTED_INSTRUMENTS)

# Ścieżki

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE = BASE_DIR / "slakh2100_flac_redux.tar.gz"

METADATA_DIR = BASE_DIR / "metadata"
SOUND_DIR = BASE_DIR / "clean_sounds"

TRACK_DIR = BASE_DIR / "tracks"
SEGMENTS_DIR = BASE_DIR / "generated_segments"

INSTRUMENTS_CSV = BASE_DIR / "instrumenty.csv"
SEGMENTS_CSV = SEGMENTS_DIR / "instrumenty.csv"

COMBINATIONS_DIR = BASE_DIR / "generated_combinations"
SUMMARY_CSV = COMBINATIONS_DIR / "plan_generowania_licznosci.csv"

PLAN_CSV = COMBINATIONS_DIR / "plan_generowania.csv"
LABELS_CSV = COMBINATIONS_DIR / "etykiety.csv"

NOISE_DIR = COMBINATIONS_DIR if NOISE_LEVEL == 0 else BASE_DIR / "noisy_sounds"

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# 1. Ekstrakcja metadanych

def extract_metadata():
    print("Ekstrakcja metadanych.")

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)

            if member.isfile() and path.name == "metadata.yaml":
                destination = METADATA_DIR.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)

                with archive.extractfile(member) as source:
                    destination.write_bytes(source.read())

    print("Ekstrakcja metadanych zakończona.")


# 2. Wyszukiwanie utworów zawierających wybrane instrumenty

def find_matched_tracks(metadata_files):
    matched_tracks = []

    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}

        stems = metadata.get("stems", {})
        matches = []

        for stem_id, stem_info in stems.items():
            instrument = stem_info.get("midi_program_name")

            if instrument not in SELECTED_INSTRUMENTS:
                continue

            matches.append({
                "stem_id": stem_id,
                "instrument": instrument,
                "is_drum": stem_info.get("is_drum", False),
                "metadata_path": metadata_path,
                "audio_dir": metadata.get("audio_dir"),
            })

        if matches:
            matched_tracks.append({
                "metadata_path": metadata_path,
                "metadata": metadata,
                "matches": matches,
            })

    print("Liczba plików metadata.yaml:", len(metadata_files))
    print("Utwory zawierające wybrane instrumenty:", len(matched_tracks))
    print(
        "Liczba pasujących ścieżek:",
        sum(len(track["matches"]) for track in matched_tracks)
    )

    return matched_tracks


# 3. Kopiowanie wybranych plików FLAC

def copy_selected_audio(matched_tracks):
    print("Rozpoczęto kopiowanie plików.")

    targets = {}

    for track in matched_tracks:
        metadata_path = track["metadata_path"]
        relative_dir = metadata_path.parent.relative_to(METADATA_DIR)

        for match in track["matches"]:
            instrument = match["instrument"]

            if instrument not in SELECTED_INSTRUMENTS:
                continue

            stem_id = match["stem_id"]

            archive_path = (
                f"{relative_dir.as_posix()}/stems/{stem_id}.flac"
            ).lower()

            targets[archive_path] = {
                "relative_dir": relative_dir,
                "stem_id": stem_id,
                "instrument": instrument,
            }

    found = set()

    if targets:
        with tarfile.open(ARCHIVE, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                archive_name = member.name.replace("\\", "/").removeprefix("./")
                key = archive_name.lower()

                if key not in targets or key in found:
                    continue

                target = targets[key]
                relative_dir = target["relative_dir"]
                stem_id = target["stem_id"]
                instrument = target["instrument"]

                output_dir = SOUND_DIR / relative_dir / "stems"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"{stem_id}.flac"

                source = tar.extractfile(member)

                if source is None:
                    continue

                with source, output_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

                found.add(key)

                percent = len(found) / len(targets) * 100

                print(
                    f"POSTĘP: {len(found)}/{len(targets)} "
                    f"({percent:.1f}%) | {instrument}"
                )

    print("Zakończono ekstrakcję plików.")


# 4. Tworzenie instrumenty.csv

def create_instruments_csv(metadata_files):
    csv_rows = []

    for metadata_path in metadata_files:

        track_dir = next(
            (
                parent for parent in metadata_path.parents
                if re.fullmatch(r"Track\d+", parent.name)
            ),
            None,
        )

        if track_dir is None:
            continue

        track_name = track_dir.name

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}

        stems = metadata.get("stems", {})

        instrument_stems = {
            instrument: []
            for instrument in SELECTED_INSTRUMENTS
        }

        for stem_id, stem_info in stems.items():
            instrument = stem_info.get("midi_program_name")

            if instrument in instrument_stems:
                instrument_stems[instrument].append(stem_id)

        row = {"Track": track_name}

        for instrument in SELECTED_INSTRUMENTS:
            row[instrument] = int(bool(instrument_stems[instrument]))

        csv_rows.append(row)

    fieldnames = ["Track"] + instruments

    with INSTRUMENTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Liczba utworów w CSV: {len(csv_rows)}")
    print(f"Plik CSV: {INSTRUMENTS_CSV}")

# 5. Segmentacja plików audio

def segment_audio(metadata_files):
    csv_columns = [
        "track_name",
        "segment_start_s",
        "segment_end_s",
        "audio_file",
    ] + instruments

    csv_rows = []

    for metadata_path in metadata_files:

        track_dir = next(
            (
                parent for parent in metadata_path.parents
                if re.fullmatch(r"Track\d+", parent.name)
            ),
            None,
        )

        if track_dir is None:
            continue

        track_name = track_dir.name

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}

        stems = metadata.get("stems", {})

        relative_dir = metadata_path.parent.relative_to(METADATA_DIR)
        stems_dir = SOUND_DIR / relative_dir / "stems"

        instrument_stems = {}

        for instrument in instruments:
            stem_id = next(
                (
                    stem_id
                    for stem_id, info in stems.items()
                    if info.get("midi_program_name") == instrument
                    and info.get("audio_rendered", True)
                ),
                None,
            )

            if stem_id is None:
                continue

            audio_path = stems_dir / f"{stem_id}.flac"

            if not audio_path.is_file():
                print(f"[BRAK FLAC] {track_name}: {audio_path}")
                continue

            audio, sr = sf.read(audio_path, dtype="float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            segment_samples = int(SEGMENT_LENGTH * sr)
            segment_data = []

            for start_sample in range(0, len(audio), segment_samples):
                end_sample = min(
                    start_sample + segment_samples,
                    len(audio),
                )

                segment = audio[start_sample:end_sample]
                segment_mean = np.mean(segment ** 2)
                is_silent = np.all(segment == 0)

                segment_data.append(
                    {
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "mean_energy": segment_mean,
                        "is_silent": is_silent,
                    }
                )

            non_silent_energies = [
                segment["mean_energy"]
                for segment in segment_data
                if not segment["is_silent"]
            ]

            overall_mean = (
                np.mean(non_silent_energies)
                if non_silent_energies
                else 0.0
            )

            threshold = ENERGY_THRESHOLD * overall_mean

            instrument_stems[instrument] = {
                "audio": audio,
                "sr": sr,
                "threshold": threshold,
            }

        if not instrument_stems:
            print(f"Pominięto {track_name}: brak dostępnych plików FLAC.")
            continue

        track_output_dir = SEGMENTS_DIR / track_name
        track_output_dir.mkdir(parents=True, exist_ok=True)

        all_stems = list(instrument_stems.values())

        sample_rates = {stem["sr"] for stem in all_stems}

        if len(sample_rates) != 1:
            print(
                f"Pominięto {track_name}: "
                "instrumenty mają różne częstotliwości próbkowania."
            )
            continue

        sr = all_stems[0]["sr"]
        segment_samples = int(SEGMENT_LENGTH * sr)

        track_samples = min(
            len(stem["audio"])
            for stem in all_stems
        )

        for start_sample in range(
            0,
            track_samples,
            segment_samples,
        ):
            end_sample = min(
                start_sample + segment_samples,
                track_samples,
            )

            active_instruments = []
            instrument_labels = {
                instrument: 0
                for instrument in instruments
            }

            for instrument, stem in instrument_stems.items():
                segment = stem["audio"][start_sample:end_sample].copy()

                if len(segment) == 0 or np.all(segment == 0):
                    continue

                segment_mean = np.mean(segment ** 2)

                if segment_mean >= stem["threshold"]:
                    instrument_labels[instrument] = 1
                    active_instruments.append((instrument, segment))

            if not active_instruments:
                continue

            start_seconds = start_sample / sr
            end_seconds = end_sample / sr

            segment_dir = track_output_dir / str(int(start_seconds))
            segment_dir.mkdir(parents=True, exist_ok=True)

            for instrument, segment in active_instruments:
                output_path = segment_dir / f"{instrument}.flac"

                if output_path.exists():
                    continue

                peak = np.max(np.abs(segment))

                if peak > 1.0:
                    segment /= peak

                sf.write(
                    output_path,
                    segment,
                    sr,
                    format="FLAC",
                    subtype="PCM_16",
                )

            first_instrument = active_instruments[0][0]

            audio_filename = (
                f"{track_name}/{int(start_seconds)}/{first_instrument}.flac"
            )

            csv_rows.append({
                "track_name": track_name,
                "segment_start_s": start_seconds,
                "segment_end_s": end_seconds,
                "audio_file": audio_filename,
                **instrument_labels,
            })

        print(f"Przetworzono: {track_name}")

    # Zapis CSV z segmentami

    with SEGMENTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Liczba zapisanych segmentów: {len(csv_rows)}")
    print(f"Zapisano plik CSV: {SEGMENTS_CSV}")

# 6. Tworzenie planu generowania kombinacji

def create_generation_plan():

    COMBINATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Zbierz fragmenty źródłowe dla każdej kombinacji instrumentów.
    examples_by_combo = defaultdict(list)

    track_dirs = sorted(
        path for path in SEGMENTS_DIR.iterdir()
        if path.is_dir() and path.name.startswith("Track")
    )

    for track_dir in track_dirs:
        track_name = track_dir.name

        segment_dirs = sorted(
            (
                path for path in track_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )

        for segment_dir in segment_dirs:
            segment_start_s = int(segment_dir.name)

            instrument_files = {
                path.stem: path
                for path in segment_dir.glob("*.flac")
            }

            present_instruments = sorted(instrument_files)

            if not present_instruments:
                continue

            # Każdy niepusty podzbiór obecnych instrumentów.
            for size in range(1, len(present_instruments) + 1):
                for combo in combinations(present_instruments, size):
                    examples_by_combo[combo].append({
                        "track_name": track_name,
                        "segment_start_s": segment_start_s,
                        "source_files": [
                            str(
                                instrument_files[instrument].relative_to(
                                    SEGMENTS_DIR
                                )
                            )
                            for instrument in combo
                        ],
                    })

    # Przygotuj plan
    plan_rows = []
    category_summary = []

    all_instruments = sorted({
        instrument
        for combo in examples_by_combo
        for instrument in combo
    })

    for combo in sorted(examples_by_combo, key=lambda c: (len(c), c)):
        # Deduplikacja po utworze i sekundzie początku.
        unique_examples = {}

        for example in examples_by_combo[combo]:
            key = (
                example["track_name"],
                example["segment_start_s"],
            )
            unique_examples[key] = example

        examples = list(unique_examples.values())
        random.shuffle(examples)

        selected_examples = examples[:TARGET_PER_CATEGORY]

        category_name = " + ".join(combo)

        category_summary.append({
            "kombinacja": category_name,
            "dostepne_fragmenty": len(examples),
            "zaplanowane_przyklady": len(selected_examples),
        })

        for example in selected_examples:
            track_name = example["track_name"]
            segment_start_s = example["segment_start_s"]

            # Unikalny numer kombinacji dla danego utworu i fragmentu.
            output_index = len(plan_rows)
            filename = (
                f"{track_name}_{segment_start_s}_{output_index}.flac"
            )

            row = {
                "output_file": filename,
                "track_name": track_name,
                "segment_start_s": segment_start_s,
                "combination": category_name,
                "combination_size": len(combo),
                "source_files": " | ".join(example["source_files"]),
            }

            for instrument in all_instruments:
                row[instrument] = int(instrument in combo)

            plan_rows.append(row)

    # Zapis planu generowania.
    base_columns = [
        "output_file",
        "track_name",
        "segment_start_s",
        "combination",
        "combination_size",
        "source_files",
    ]

    csv_columns = base_columns + all_instruments

    with open(PLAN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(plan_rows)

    # Zapis liczności każdej kategorii.
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kombinacja",
                "dostepne_fragmenty",
                "zaplanowane_przyklady",
            ],
        )
        writer.writeheader()
        writer.writerows(category_summary)

    print(f"Liczba kategorii obecnych w danych: {len(category_summary)}")
    print(f"Liczba zaplanowanych nagrań: {len(plan_rows)}")
    print(f"Plan zapisano w: {PLAN_CSV}")
    print(f"Liczności zapisano w: {SUMMARY_CSV}")

# 7. Generowanie nagrań i etykiet

def generate_combinations_audio():

    COMBINATIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PLAN_CSV, "r", newline="", encoding="utf-8-sig") as f:
        plan_rows = list(csv.DictReader(f))

    metadata_columns = {
        "output_file",
        "track_name",
        "segment_start_s",
        "combination",
        "combination_size",
        "source_files",
    }

    instruments = [
        column
        for column in (plan_rows[0].keys() if plan_rows else [])
        if column not in metadata_columns
    ]

    label_rows = []

    for row in plan_rows:
        output_path = COMBINATIONS_DIR / row["output_file"]

        source_files = [
            SEGMENTS_DIR / path.strip()
            for path in row["source_files"].split("|")
            if path.strip()
        ]

        audio_parts = []
        sr = None

        for source_path in source_files:
            audio, sr = sf.read(source_path, dtype="float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio_parts.append(audio)

        max_length = max(len(audio) for audio in audio_parts)
        mix = np.zeros(max_length, dtype=np.float32)

        for audio in audio_parts:
            mix[:len(audio)] += audio

        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak

        sf.write(
            output_path,
            mix,
            sr,
            format="FLAC",
            subtype="PCM_16",
        )

        label_row = {"audio_file": row["output_file"]}

        for instrument in instruments:
            label_row[instrument] = int(row[instrument] == "1")

        label_rows.append(label_row)
        print(f"Wygenerowano: {row['output_file']}")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_file"] + instruments,
        )
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"Wygenerowano nagrań: {len(label_rows)}")
    print(f"Zapisano etykiety: {LABELS_CSV}")

# 8. Dodawanie białego szumu


def add_white_noise():

    NOISE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dodawanie białego szumu do wygenerowanych fragmentów

    audio_files = sorted(COMBINATIONS_DIR.glob("*.flac"))

    for audio_path in audio_files:
        audio, sr = sf.read(audio_path, dtype="float32")

        noise = np.random.normal(
            0,
            NOISE_LEVEL,
            size=audio.shape
        ).astype(np.float32)

        noisy_audio = np.clip(audio + noise, -1.0, 1.0)

        output_path = NOISE_DIR / audio_path.name

        sf.write(
            output_path,
            noisy_audio,
            sr,
            format="FLAC",
            subtype="PCM_16"
        )

    # 2. Generowanie nagrań zawierających wyłącznie biały szum

    if not audio_files:
        raise FileNotFoundError("Brak wygenerowanych plików FLAC.")

    sr = sf.info(audio_files[0]).samplerate
    num_samples = int(SEGMENT_LENGTH * sr)

    for i in range(1, TARGET_PER_CATEGORY + 1):
        white_noise = np.random.normal(
            0,
            NOISE_LEVEL,
            size=num_samples
        ).astype(np.float32)

        output_path = NOISE_DIR / f"Track00000_{i:05d}.flac"

        sf.write(
            output_path,
            white_noise,
            sr,
            format="FLAC",
            subtype="PCM_16"
        )

    print(f"Zapisano {len(audio_files)} zaszumionych nagrań.")
    print(f"Wygenerowano {TARGET_PER_CATEGORY} nagrań z białym szumem.")

# 9. Tworzenie zbioru danych

def create_dataset():

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    with open(PLAN_CSV, "r", newline="", encoding="utf-8-sig") as f:
        plan_rows = list(csv.DictReader(f))

    with open(LABELS_CSV, "r", newline="", encoding="utf-8-sig") as f:
        label_reader = csv.DictReader(f)
        label_rows = list(label_reader)
        label_columns = label_reader.fieldnames

    labels_by_file = {
        row["audio_file"]: row
        for row in label_rows
    }

    groups = {}

    for row in plan_rows:
        combination = row["combination"]
        groups.setdefault(combination, []).append(row)

    train_rows = []
    validation_rows = []
    test_rows = []

    for combination, rows in groups.items():
        indices = list(range(len(rows)))
        random.shuffle(indices)

        shuffled_rows = [rows[i] for i in indices]

        n = len(shuffled_rows)
        n_train = int(round(n * TRAIN_RATIO))
        n_validation = int(round(n * VALIDATION_RATIO))

        train_rows.extend(shuffled_rows[:n_train])

        validation_rows.extend(
            shuffled_rows[n_train:n_train + n_validation]
        )

        test_rows.extend(
            shuffled_rows[n_train + n_validation:]
        )

    random.shuffle(train_rows)
    random.shuffle(validation_rows)
    random.shuffle(test_rows)

    datasets = [
        ("train", TRAIN_DIR, train_rows),
        ("validation", VALIDATION_DIR, validation_rows),
        ("test", TEST_DIR, test_rows),
    ]

    all_label_rows = []

    for dataset_name, dataset_dir, rows in datasets:

        for row in rows:
            audio_file = row["output_file"]

            source_path = NOISE_DIR / audio_file
            destination_path = dataset_dir / audio_file

            destination_path.write_bytes(source_path.read_bytes())

            label = labels_by_file[audio_file].copy()

            all_label_rows.append({
                "dataset": dataset_name,
                **label,
            })

    with open(
        DATASET_DIR / "etykiety.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["dataset"] + label_columns,
        )

        writer.writeheader()
        writer.writerows(all_label_rows)

    total = len(plan_rows)

    print(f"Wszystkie nagrania: {total}")

    for dataset_name, dataset_dir, rows in datasets:

        percentage = len(rows) / total * 100 if total else 0

        print(
            f"{dataset_name}: {len(rows)} "
            f"({percentage:.2f}%)"
        )

        print(f"Folder: {dataset_dir}")

    print(f"Plik etykiet: {DATASET_DIR / 'etykiety.csv'}")


# Kontrola parametrów poprzedniego uruchomienia
 
STATE_CSV = BASE_DIR / "pipeline_state.csv"

def load_pipeline_state():
    if not STATE_CSV.exists():
        return {}

    with STATE_CSV.open("r", newline="", encoding="utf-8") as f:
        return {
            row["stage"]: row["signature"]
            for row in csv.DictReader(f)
        }


def save_pipeline_state(state):
    with STATE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "signature"],
        )
        writer.writeheader()

        for stage, signature in state.items():
            writer.writerow({
                "stage": stage,
                "signature": signature,
            })


def stage_changed(state, stage, parameters):
    signature = repr(parameters)
    return state.get(stage) != signature


def remove_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)

# 


def generate_autio(
    SELECTED_INSTRUMENTS,
    SEGMENT_LENGTH,
    ENERGY_THRESHOLD,
    NOISE_LEVEL,
    TARGET_PER_CATEGORY,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
):
    global instruments
    global NOISE_DIR

    globals()["SELECTED_INSTRUMENTS"] = set(SELECTED_INSTRUMENTS)
    globals()["SEGMENT_LENGTH"] = SEGMENT_LENGTH
    globals()["ENERGY_THRESHOLD"] = ENERGY_THRESHOLD
    globals()["NOISE_LEVEL"] = NOISE_LEVEL
    globals()["TARGET_PER_CATEGORY"] = TARGET_PER_CATEGORY
    globals()["TRAIN_RATIO"] = TRAIN_RATIO
    globals()["VALIDATION_RATIO"] = VALIDATION_RATIO
    globals()["TEST_RATIO"] = TEST_RATIO

    instruments = sorted(SELECTED_INSTRUMENTS)

    NOISE_DIR = (
        COMBINATIONS_DIR
        if NOISE_LEVEL == 0
        else BASE_DIR / "noisy_sounds"
    )

    random.seed(2137)

    state = load_pipeline_state()

    # 1. Metadane

    metadata_rebuilt = False

    if not METADATA_DIR.exists() or not any(
        METADATA_DIR.rglob("metadata.yaml")
    ):
        if not ARCHIVE.exists():
            raise FileNotFoundError(f"Brakuje archiwum: {ARCHIVE}")

        remove_directory(METADATA_DIR)
        extract_metadata()
        metadata_rebuilt = True

    state["metadata"] = repr("metadata")

    metadata_files = sorted(
        METADATA_DIR.rglob("metadata.yaml")
    )

    matched_tracks = find_matched_tracks(metadata_files)

    # 2. Kopiowanie plików audio

    sound_parameters = (
        tuple(instruments),
    )

    sound_rebuilt = (
        not SOUND_DIR.exists()
        or not any(SOUND_DIR.rglob("*.flac"))
        or metadata_rebuilt
        or stage_changed(state, "sounds", sound_parameters)
    )

    if sound_rebuilt:
        if not ARCHIVE.exists():
            raise FileNotFoundError(f"Brakuje archiwum: {ARCHIVE}")

        remove_directory(SOUND_DIR)
        copy_selected_audio(matched_tracks)

    state["sounds"] = repr(sound_parameters)

    # 3. Plik instrumenty.csv

    instruments_parameters = tuple(instruments)

    if (
        not INSTRUMENTS_CSV.exists()
        or stage_changed(
            state,
            "instruments_csv",
            instruments_parameters,
        )
        or metadata_rebuilt
    ):
        create_instruments_csv(metadata_files)

    state["instruments_csv"] = repr(instruments_parameters)

    # 4. Segmentacja audio

    segments_parameters = (
        tuple(instruments),
        SEGMENT_LENGTH,
        ENERGY_THRESHOLD,
    )

    segments_rebuilt = (
        not SEGMENTS_DIR.exists()
        or not SEGMENTS_CSV.exists()
        or sound_rebuilt
        or stage_changed(
            state,
            "segments",
            segments_parameters,
        )
    )

    if segments_rebuilt:
        remove_directory(SEGMENTS_DIR)
        SEGMENTS_DIR.mkdir(parents=True)

        print("Rozpoczynam segmentację plików audio.")
        segment_audio(metadata_files)

    state["segments"] = repr(segments_parameters)

    # 5. Plan generowania i kombinacje audio

    combinations_parameters = (
        tuple(instruments),
        TARGET_PER_CATEGORY,
    )

    combinations_rebuilt = (
        not PLAN_CSV.exists()
        or not SUMMARY_CSV.exists()
        or not LABELS_CSV.exists()
        or not any(COMBINATIONS_DIR.glob("*.flac"))
        or segments_rebuilt
        or stage_changed(
            state,
            "combinations",
            combinations_parameters,
        )
    )

    if combinations_rebuilt:
        remove_directory(COMBINATIONS_DIR)
        COMBINATIONS_DIR.mkdir(parents=True)

        print("Tworzenie planu generowania...")
        create_generation_plan()

        generate_combinations_audio()

    state["combinations"] = repr(combinations_parameters)

    # 6. Dodawanie szumu

    noise_rebuilt = False

    if NOISE_LEVEL == 0:
        NOISE_DIR = COMBINATIONS_DIR

    else:
        noise_parameters = (NOISE_LEVEL,)

        combination_files = list(
            COMBINATIONS_DIR.glob("*.flac")
        )

        noise_rebuilt = (
            not NOISE_DIR.exists()
            or combinations_rebuilt
            or stage_changed(
                state,
                "noise",
                noise_parameters,
            )
            or not all(
                (NOISE_DIR / audio_path.name).exists()
                for audio_path in combination_files
            )
        )

        if noise_rebuilt:
            remove_directory(NOISE_DIR)
            add_white_noise()

        state["noise"] = repr(noise_parameters)

    # 7. Tworzenie zbioru danych

    dataset_parameters = (
        TRAIN_RATIO,
        VALIDATION_RATIO,
        TEST_RATIO,
    )

    dataset_rebuilt = (
        not DATASET_DIR.exists()
        or not TRAIN_DIR.exists()
        or not VALIDATION_DIR.exists()
        or not TEST_DIR.exists()
        or not (DATASET_DIR / "etykiety.csv").exists()
        or combinations_rebuilt
        or noise_rebuilt
        or stage_changed(
            state,
            "dataset",
            dataset_parameters,
        )
    )

    if dataset_rebuilt:
        remove_directory(DATASET_DIR)
        create_dataset()

    state["dataset"] = repr(dataset_parameters)

    save_pipeline_state(state)
