import csv
import hashlib
import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "pleasant_surprise", "sadness"]


def config():
    return json.loads((ROOT / "configs/experiment.json").read_text())


def manifest():
    with (ROOT / "data/manifests/manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def fingerprint():
    return hashlib.sha256((ROOT / "data/manifests/manifest.csv").read_bytes()).hexdigest()


def load_features():
    data = torch.load(ROOT / ".local/features.pt", map_location="cpu", weights_only=False)
    assert data["manifest_sha256"] == fingerprint(), "Feature cache is stale; regenerate it."
    assert data["preprocessing"] == {k: config()[k] for k in data["preprocessing"]}, "Preprocessing config changed."
    return data


def indices(rows, protocol, split):
    return [i for i, row in enumerate(rows) if row[f"split_{protocol}"] == split]


def audio_stats(audio, ids):
    # Only training frames contribute to normalization.
    frames = torch.cat([audio[i] for i in ids], 0)
    return frames.mean(0), frames.std(0).clamp_min(1e-5)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, ids, mean, std):
        self.data, self.ids, self.mean, self.std = data, ids, mean, std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        return (self.data["audio"][i] - self.mean) / self.std, self.data["labels"][i], i


def collate_audio(batch):
    audio, labels, ids = zip(*batch)
    return pad_sequence(audio, batch_first=True), torch.tensor([len(a) for a in audio]), torch.tensor(labels), torch.tensor(ids)
