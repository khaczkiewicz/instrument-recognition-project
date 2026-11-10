import os
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf

from tqdm import tqdm


#======================================#
# Funkcje do konstrukcji spektrogramów #
#======================================#

def signal_zero_padding(audio_data, frame_len = 1024, step_len = 512):

    audio_data = np.asarray(audio_data)
    padding_left = frame_len//2
    num_samples = len(audio_data)
    r = num_samples%step_len

    if r != 0:
        add_right = step_len - r
    else:
        add_right = 0

    padding_right = frame_len//2 + add_right


    padded_signal = np.zeros(padding_left + num_samples + padding_right, dtype=audio_data.dtype)
    padded_signal[padding_left: padding_left+num_samples] = audio_data

    return padded_signal


def signal_reflect_padding(audio_data, frame_len = 1024, step_len = 512):

    audio_data = np.asarray(audio_data)
    padding_left = frame_len//2
    num_samples = len(audio_data)
    r = num_samples%step_len

    if r != 0:
            add_right = step_len - r
    else:
        add_right = 0
    
    padding_right = frame_len//2 + add_right

    pad_left = audio_data[1:padding_left+1][::-1]
    pad_right = audio_data[-1-padding_right : -1][::-1]

    padded_signal = np.concatenate([pad_left, audio_data, pad_right])

    return padded_signal


def frame_audio(audio_data, frame_len = 1024, step_len = 512):

    audio_data = np.asarray(audio_data)
    num_of_samples = len(audio_data)
    number_frames = (num_of_samples-frame_len)//step_len + 1

    frames = np.empty((number_frames,frame_len), dtype=audio_data.dtype)
    
    for i in range(number_frames):
        start = i*step_len
        frames[i] = audio_data[start: frame_len + start]

    return frames


def hann_window(frame_len=1024):
    return np.asarray([ (0.5 - 0.5*np.cos(2*np.pi*n/ (frame_len-1))) for n in range(frame_len) ])

def create_dft_matrix(frame_len = 1024):

    # Kształt macierzy (frame_len, frame_len // 2 + 1)
    N=frame_len
    num_unique_k = N//2 + 1

    n = np.arange(N).reshape(N,1)
    k = np.arange(num_unique_k).reshape(1,num_unique_k)

    dft_matrix = np.exp(-2j*np.pi * n * k / N)
    return dft_matrix.astype(np.complex64)

def DFT(signal):
    signal = np.asarray(signal)
    N = signal.shape[-1]
    #broadcasting
    #wiersze to czas (n), a kolumny to częstotliwość (k)
    k = np.arange(N)
    n = np.arange(N).reshape(N,1)

    expr = np.exp(-2j * np.pi * k * n / N) # symetryczna macierz

    # signal (M,N) 
    return np.dot(signal, expr)

def FFT(signal):
    # Funkcja ma obsługiwać wejście (liczba_ramek x długość ramki)
    # zrobić: wektoryzacja funkcji
    signal = np.asarray(signal)
    N = len(signal)

    if N%2 > 0:
        raise ValueError('Długość sygnału powinna być potęgą 2')

    elif N <= 8:
        return DFT(signal)

    k = np.arange(N//2)
    
    even = FFT(signal[::2])
    odd = FFT(signal[1::2])

    expr = np.exp(-2j * np.pi * k  / N)

    X_first_half = even + expr*odd
    X_second_half = even - expr*odd
    return np.concat([X_first_half,X_second_half])

# Filtry melowe

def freq_to_mel(point):
    return 2595*np.log10(1 + point/700)

def mel_to_freq(point):
    return 700* (10**(point/2595) - 1)

def create_mel_filterbank(lowest_freq, highest_freq, n_mel_filters = 64, frame_len = 1024, sr = 44100):
    lowest_mel = freq_to_mel(lowest_freq)
    highest_mel = freq_to_mel(highest_freq)

    # Liczba prążków widma DFT
    n_bins = frame_len//2 +1

    mel_grid = np.linspace(lowest_mel, highest_mel, num = n_mel_filters + 2)

    # Konwersja do Hz
    herz_grid = mel_to_freq(mel_grid)

    # Musimy przybliżeć te częstotliwości do najbliższej częstotliwości ramki po zastosowaniu DFT
    nearest_DFT_bin =  np.floor((frame_len * herz_grid/ sr))

    # Inicjalizacja macierzy filtrów (513, liczba filtrów mel)
    # każdy wiersz to pojedynczy filtr Melowy o długości równej liczbie prążków DFT (513) 
    filterbanks = np.zeros((n_mel_filters, n_bins), dtype=np.float32)

    # Wypełnienie macierzy filtrami trójkątnymi
    for m in range(1, n_mel_filters + 1):
        left_point = int(nearest_DFT_bin[m-1])
        center_point = int(nearest_DFT_bin[m])
        right_point = int(nearest_DFT_bin[m+1])
        #print(left_point, center_point, right_point)

        if left_point < center_point:
            for k in range(left_point, center_point):
                #print((k-left_point)/ (center_point - left_point))
                filterbanks[m-1, k] = (k-left_point)/ (center_point - left_point)

        if center_point < right_point:
            for k in range(center_point, right_point):
                #print( (right_point + k)/(right_point - center_point) )
                filterbanks[m-1, k] = (right_point - k)/(right_point - center_point) 

    return filterbanks

#======================================#
#      Wyznaczanie spektrogramów       #
#======================================#

def compute_mel_spectrogram(audio_signal,
                            window_vector,
                            dft_matrix,
                            mel_filterbanks,
                            frame_length = 1024,
                            step_length = 512,
                            sampling_rate = 44100,
                            padding_type = 'reflect',
                            db= True):
    
    if padding_type == 'reflect':
        padded_signal = signal_reflect_padding(audio_data=audio_signal,
                                           frame_len=frame_length,
                                           step_len=step_length)
    elif padding_type == 'zero':
        padded_signal = signal_zero_padding(audio_data=audio_signal,
                                               frame_len=frame_length,
                                               step_len=step_length)
    elif padding_type == None:
        padded_signal = audio_signal

    sliced_audio = frame_audio(padded_signal, frame_len=frame_length, step_len=step_length)

    windowed_frames = sliced_audio * window_vector

    dft_frames = np.dot(windowed_frames, dft_matrix) # (M,N) @ (N,K) -> (M,K)

    spectrogram = (np.abs(dft_frames)**2) / frame_length

    mel_spectrogram = np.dot(spectrogram, mel_filterbanks.T) # (M,K) @ (B,K).T -> (M,B)
    
    if db==True:
        replace_zero = 1e-10
        mel_spectrogram = np.maximum(mel_spectrogram, replace_zero)
        mel_spectrogram = 10.0 * np.log10(mel_spectrogram)
        mel_spectrogram = mel_spectrogram - np.max(mel_spectrogram)
        mel_spectrogram = np.maximum(mel_spectrogram, -80.0)

    return mel_spectrogram.T.astype(np.float32) # Transpozycja na (B,M) - czas na OX
        


#======================================#
#      Przetwarzanie zbioru danych     #
#======================================#

def generate_spectrograms_from_dataset(dataset_dir,
                                       output_dir,
                                       sample_rate = 44100,
                                       expected_duration = 4.0,
                                       overwrite = False,

                                       frame_length = 1024,
                                       step_length = 512,
                                       padding_type = 'reflect',
                                       window_type = 'hann',
                                       lowest_freq = 20,
                                       highest_freq = 22050,
                                       n_mel_banks = 80,
                                       db= True
                                       ):

    """Przetwarza katalogi plików .wav i .flac na spektrogramy .npy"""

    # Stworzenie folderu (nazwa to parametry spektrogramu)
    spectrogram_params_folder_name = (f'mels{n_mel_banks}_frame{frame_length}_step{step_length}_window{window_type}_sr{sample_rate}_{int(expected_duration)}s')

    dataset_path = Path(dataset_dir)

    output_path = Path(output_dir)/spectrogram_params_folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    expected_num_of_samples = int(sample_rate*expected_duration)
    expected_filetype = {'.wav', '.flac'}

    audio_files = sorted([audio_file for audio_file in dataset_path.rglob('*') 
                   if audio_file.suffix.lower() in expected_filetype])

    print(f'Katalog wejsciowy: {dataset_path}')
    print(f'Katalog docelowy: {output_path}')
    print(f'Znaleziono {len(audio_files)} plików audio')

    ########################
    # Wyznaczenie macierzy #

    if window_type == 'hann':
        window_vector = hann_window(frame_len=frame_length).astype(np.float32)
    else:
        window_vector = np.ones(frame_length, dtype=np.float32)

    dft_matrix = create_dft_matrix(frame_len=frame_length)

    mel_filterbanks = create_mel_filterbank(
        lowest_freq=lowest_freq,
        highest_freq=highest_freq,
        n_mel_filters=n_mel_banks,
        frame_len=frame_length,
        sr=sample_rate
    ).astype(np.float32)

    #####################


    metadata=[]
    excluded_files = []

    for file_path in tqdm(audio_files, desc='Przetwarzanie plików'):
        try:
            # Podaje dalsze podfoldery pliku od zadanego w argumencie
            subpath = file_path.relative_to(dataset_path)
            target_file = output_path/subpath.with_suffix('.npy') # Dodajemy rozszerzenie do nazwy pliku

            metadata_entry = {
                'file_path': str(subpath.with_suffix('.npy')),
                'set': subpath.parts[0] if len(subpath.parts) > 1 else "not_defined", # Wskazuje na zbiór test/ train/ val
                'stem_name': file_path.stem
            }

            # Sprawdzamy czy plik już istnieje
            if target_file.exists() and overwrite == False:
                metadata.append(metadata_entry)
                continue

            # Sprawdzamy czy docelowy katalog istnieje
            target_file.parent.mkdir(exist_ok=True, parents=True)

            # Wczytanie pliku audio
            audio, sr = sf.read(file_path, dtype='float32')

            # Konwersja do mono w przypadku plików stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sr != sample_rate:
                raise ValueError(f'Niezgodna wartość częstotliwości próbkowania: {sr} Hz w pliku {file_path.name} (oczekiwane {sample_rate} Hz).')

            # Sprawdzenie długości sygnału
            if len(audio) < expected_num_of_samples:
                audio = np.pad(audio, (0, expected_num_of_samples - len(audio)) ) #Zmodyfikować funkcję padding_reflect()
            elif len(audio) > expected_num_of_samples:
                audio = audio[:expected_num_of_samples]

            # Wyznaczenie log-mel spektrogramu

            # Macierz filtrów melowych, FFT, funkcję okna można wyznaczyć tylko raz
            log_mel_spectrogram = compute_mel_spectrogram(audio_signal=audio,
                                                          window_vector=window_vector,
                                                          dft_matrix=dft_matrix,
                                                          mel_filterbanks=mel_filterbanks,
                                                          frame_length=frame_length,
                                                          step_length=step_length,
                                                          sampling_rate=sample_rate,
                                                          padding_type=padding_type,
                                                          db=db
                                                        )

            # Zapis do pliku .npy
            np.save(target_file, log_mel_spectrogram.astype(np.float32))

            # Zapis w metadanych
            metadata.append(metadata_entry)

        except Exception as error_log:
            excluded_files.append( (str(file_path), str(error_log)) )

    # Zapis metadanych do csv
    df_meta = pd.DataFrame(metadata)
    csv_file_path = output_path/ 'metadata.csv'
    df_meta.to_csv(csv_file_path, index=False)

    print("## Koniec ##")
    print(f"Zapisano {len(df_meta)} spektrogramów oraz plik csv.")

    if excluded_files:
        print(f'UWAGA: pominięto {len(excluded_files)} plików')
        for path, error in excluded_files[:5]: # pierwsze 5 plików z opisem błędu
            print(f" > {path}: {error}")

    return output_path
    