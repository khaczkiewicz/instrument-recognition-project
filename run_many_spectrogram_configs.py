from spectrograms_pipeline import generate_spectrograms_from_dataset

# Zestawy parametrów do przetestowania
configs = [
    {"n_mel_banks": 128, "frame_length": 1024, "step_length": 512},
    {"n_mel_banks": 80, "frame_length": 2048, "step_length": 512},
    {"n_mel_banks": 128, "frame_length": 2048, "step_length": 512},
    {"n_mel_banks": 128, "frame_length": 2048, "step_length": 1024},
    {"n_mel_banks": 128, "frame_length": 4096, "step_length": 1024}
]

for config in configs:
    generate_spectrograms_from_dataset(
        dataset_dir="data/slakh_combination_100",
        output_dir="spectrograms", 
        sample_rate=44100,
        expected_duration=4.0,
        overwrite=False,
        **config
    )