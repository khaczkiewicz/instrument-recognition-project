
import torch
import torch.nn as nn
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