
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTConfig, ASTForAudioClassification


class InstrumentAST(nn.Module):
    """
    Architektura Audio Spectrogram Transformer (AST) przeznaczona do
    wieloetykietowego rozpoznawania obecności 7 instrumentów muzycznych
    na podstawie log-mel spektrogramów.

    Wejście:
        (batch_size, 1, mel_bins, czas)

    Wyjście:
        (batch_size, 7)
    """

    def __init__(self, num_mel_bins, max_length, dropout=0.1):
        super().__init__()

        # Konfiguracja architektury AST.
        config = ASTConfig(
            num_mel_bins=num_mel_bins,
            max_length=max_length,
            num_labels=7,
            patch_size=16,
            frequency_stride=10,
            time_stride=10,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            problem_type="multi_label_classification"
        )

        # Utworzenie modelu AST z losowo inicjalizowanymi wagami.
        self.model = ASTForAudioClassification(config)

    def forward(self, x):
        # Zmiana formatu na (batch_size, czas, mel_bins).
        x = x.squeeze(1).transpose(1, 2)

        # Przejście przez AST.
        outputs = self.model(input_values=x)

        return outputs.logits


class InstrumentASTPretrained(nn.Module):
    """
    Audio Spectrogram Transformer z wagami wstępnie wytrenowanymi
    na zbiorze AudioSet.

    Warstwa klasyfikacyjna jest dostosowana do 7 instrumentów.

    Wejście:
        (batch_size, 1, mel_bins, czas)

    Wyjście:
        (batch_size, 7)
    """

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    MAX_LENGTH = 1024

    def __init__(self, dropout=0.1):
        super().__init__()

        # Załadowanie wag pretrained i dostosowanie klasyfikatora.
        self.model = ASTForAudioClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=7,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

        # Dostosowanie dropout do eksperymentu.
        self.model.config.hidden_dropout_prob = dropout
        self.model.config.attention_probs_dropout_prob = dropout

        self.model.audio_spectrogram_transformer.config.hidden_dropout_prob = dropout
        self.model.audio_spectrogram_transformer.config.attention_probs_dropout_prob = dropout

    def forward(self, x):
        # Zmiana formatu na (batch_size, czas, mel_bins).
        x = x.squeeze(1).transpose(1, 2)

        # Uzupełnienie sekwencji do długości wymaganej przez pretrained AST.
        if x.shape[1] > self.MAX_LENGTH:
            x = x[:, :self.MAX_LENGTH, :]
        elif x.shape[1] < self.MAX_LENGTH:
            x = F.pad(x, (0, 0, 0, self.MAX_LENGTH - x.shape[1]))

        # Przejście przez AST.
        outputs = self.model(input_values=x)

        return outputs.logits