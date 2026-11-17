"""Create reusable features; statistical normalization is fitted later per split."""
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tess_emotion.data import ROOT, config, manifest, fingerprint


def mel_bank(rate, n_fft, bins):
    melmax = 2595 * np.log10(1 + rate / 2 / 700)
    points = 700 * (10 ** (np.linspace(0, melmax, bins + 2) / 2595) - 1)
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    bank = np.maximum(0, np.minimum((freqs[:, None] - points[:-2]) / (points[1:-1] - points[:-2]),
                                    (points[2:] - freqs[:, None]) / (points[2:] - points[1:-1])))
    return torch.tensor(bank, dtype=torch.float32)


def trim(audio, rate, threshold_db, margin_ms):
    frame, hop = int(.02 * rate), int(.01 * rate)
    starts = np.arange(0, max(1, len(audio) - frame + 1), hop)
    energy = np.array([np.sqrt(np.mean(audio[s:s + frame] ** 2)) for s in starts])
    active = np.flatnonzero(energy >= energy.max() * 10 ** (-threshold_db / 20))
    margin = int(margin_ms * rate / 1000)
    a = max(0, int(starts[active[0]]) - margin)
    b = min(len(audio), int(starts[active[-1]]) + frame + margin)
    return audio[a:b].copy(), a / rate, b / rate


def main():
    cfg = config()
    torch.set_num_threads(cfg["threads"])
    rows = manifest()
    bank = mel_bank(cfg["sample_rate"], cfg["n_fft"], cfg["n_mels"])
    audio, trim_info = [], []
    window = torch.hann_window(cfg["win_length"])
    for i, row in enumerate(rows):
        x, rate = sf.read(ROOT / row["audio_path"], dtype="float32")
        div = math.gcd(rate, cfg["sample_rate"])
        x = resample_poly(x, cfg["sample_rate"] // div, rate // div).astype(np.float32)
        x, start, end = trim(x, cfg["sample_rate"], cfg["trim_db"], cfg["trim_margin_ms"])
        spectrum = torch.stft(torch.from_numpy(x), cfg["n_fft"], cfg["hop_length"], cfg["win_length"], window,
                              return_complex=True).abs().square().T
        features = torch.log((spectrum @ bank).clamp_min(1e-8))
        assert torch.isfinite(features).all()
        audio.append(features)
        trim_info.append({"id": row["id"], "start_s": start, "end_s": end, "frames": len(features)})
        if i % 500 == 0:
            print(f"Audio features {i}/{len(rows)}", flush=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"], revision=cfg["text_revision"])
    model = AutoModel.from_pretrained(cfg["text_model"], revision=cfg["text_revision"]).eval()
    sentences = sorted({row["transcript"] for row in rows})
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(sentences), 32):
            inputs = tokenizer(sentences[start:start + 32], return_tensors="pt", padding=True, truncation=True)
            output = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].clone().bool()
            # Exclude special tokens and padding from contextual mean pooling.
            mask &= inputs["input_ids"] != tokenizer.cls_token_id
            mask &= inputs["input_ids"] != tokenizer.sep_token_id
            vectors.append((output * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
    lookup = dict(zip(sentences, torch.cat(vectors)))
    preprocessing = {k: cfg[k] for k in ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels", "trim_db", "trim_margin_ms", "text_model", "text_revision")}
    data = {"audio": audio, "text": torch.stack([lookup[r["transcript"]] for r in rows]),
            "labels": [r["label"] for r in rows], "ids": [r["id"] for r in rows],
            "manifest_sha256": fingerprint(), "preprocessing": preprocessing}
    torch.save(data, ROOT / ".local/features.pt")
    (ROOT / "Results/preprocessing.json").write_text(json.dumps({"config": preprocessing, "recordings": trim_info}, indent=2) + "\n")
    print("Saved features:", len(audio), "audio sequences and", len(lookup), "unique text embeddings", flush=True)


if __name__ == "__main__":
    main()
