import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # LayerNorm is per frame, avoiding normalization over padded time steps.
        self.conv1 = nn.Conv1d(64, 64, 5, stride=2, padding=2)
        self.norm1 = nn.LayerNorm(64)
        self.conv2 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(64)
        self.gru = nn.GRU(64, 48, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(96, 1)
        self.drop = nn.Dropout(.15)

    def forward(self, x, lengths):
        # Mask after each convolution so padded activations cannot enter the next.
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        lengths = (lengths + 1) // 2
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        x = self.drop(torch.relu(self.norm1(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        lengths = (lengths + 1) // 2
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        x = self.drop(torch.relu(self.norm2(x))) * mask[..., None]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.gru(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        scores = self.attention(x).squeeze(-1).masked_fill(~mask, -1e9)
        return (x * scores.softmax(1)[..., None]).sum(1)


class Head(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(.25))
        self.classifier = nn.Linear(hidden, 7)

    def forward(self, x, return_embedding=False):
        embedding = self.projection(x)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeechEncoder()
        self.head = Head(96)

    def forward(self, x, lengths):
        return self.head(self.encoder(x, lengths))
