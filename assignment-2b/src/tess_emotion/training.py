"""Train using training/validation only; test evaluation is a separate command."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ROOT, LABELS, config, manifest, fingerprint, load_features, indices, AudioDataset, audio_stats, collate_audio
from .networks import SpeechModel, Head


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config()["threads"])


def checkpoint_path(protocol, seed, pipeline, tag=""):
    return ROOT / "checkpoints" / (protocol + tag) / f"seed-{seed}" / f"{pipeline}.pt"


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["manifest_sha256"] == fingerprint(), "Checkpoint uses another dataset/split."
    return ckpt


def speech_embeddings(data, protocol, seed, tag=""):
    path = checkpoint_path(protocol, seed, "speech", tag)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = ROOT / ".local" / f"speech-embeddings-{protocol}{tag}-{seed}.pt"
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved["checkpoint_sha256"] == digest:
            return saved["embeddings"]
    ckpt = load_checkpoint(path)
    model = SpeechModel().eval()
    model.load_state_dict(ckpt["state_dict"])
    dataset = AudioDataset(data, list(range(len(data["audio"]))), ckpt["mean"], ckpt["std"])
    loader = DataLoader(dataset, batch_size=config()["batch_size"], collate_fn=collate_audio)
    output = []
    with torch.inference_mode():
        for x, lengths, _, _ in loader:
            output.append(model.encoder(x, lengths))
    embeddings = torch.cat(output)
    torch.save({"checkpoint_sha256": digest, "embeddings": embeddings}, cache)
    return embeddings


def raw_features(data, pipeline, protocol, seed, tag=""):
    if pipeline == "text":
        return data["text"].clone()
    speech = speech_embeddings(data, protocol, seed, tag)
    if pipeline == "speech_control":
        return speech
    assert pipeline == "fusion"
    return torch.cat([speech, data["text"]], 1)


def head_dimensions(pipeline):
    # The control has 55,855 parameters; the fusion head has 55,815.
    return {"text": (768, 64), "fusion": (864, 64), "speech_control": (96, 537)}[pipeline]


def metrics(y, predictions):
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", labels=list(range(7)), zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions))}


def run_epoch(model, loader, pipeline, optimizer=None):
    model.train(optimizer is not None)
    total_loss, ys, ps = 0., [], []
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for batch in loader:
            if pipeline == "speech":
                x, lengths, y, _ = batch
                logits = model(x, lengths)
            else:
                x, y = batch
                logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            total_loss += loss.item() * len(y)
            ys.extend(y.tolist())
            ps.extend(logits.argmax(1).tolist())
    return {"loss": total_loss / len(ys), **metrics(ys, ps)}


def train(pipeline, protocol, seed, max_epochs=None, tag=""):
    seed_everything(seed)
    cfg, rows, data = config(), manifest(), load_features()
    train_ids, val_ids = indices(rows, protocol, "train"), indices(rows, protocol, "validation")
    if pipeline == "speech":
        mean, std = audio_stats(data["audio"], train_ids)
        train_set = AudioDataset(data, train_ids, mean, std)
        val_set = AudioDataset(data, val_ids, mean, std)
        collate = collate_audio
        model = SpeechModel()
    else:
        raw = raw_features(data, pipeline, protocol, seed, tag)
        mean, std = raw[train_ids].mean(0), raw[train_ids].std(0).clamp_min(1e-5)
        x, y = (raw - mean) / std, torch.tensor(data["labels"])
        train_set = TensorDataset(x[train_ids], y[train_ids])
        val_set = TensorDataset(x[val_ids], y[val_ids])
        collate = None
        model = Head(*head_dimensions(pipeline))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum = max_epochs or cfg["speech_epochs" if pipeline == "speech" else "head_epochs"]
    history, best, stale, best_state = [], (-1., -float("inf")), 0, None
    started = time.perf_counter()
    for epoch in range(1, maximum + 1):
        clock = time.perf_counter()
        training = run_epoch(model, train_loader, pipeline, optimizer)
        validation = run_epoch(model, val_loader, pipeline)
        entry = {"epoch": epoch, "train": training, "validation": validation, "seconds": time.perf_counter() - clock}
        history.append(entry)
        score = (validation["macro_f1"], -validation["loss"])
        if score > best:
            best, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            stale += 1
        print(json.dumps({"pipeline": pipeline, "protocol": protocol, "seed": seed, **entry}), flush=True)
        if stale >= cfg["patience"]:
            break
    path = checkpoint_path(protocol, seed, pipeline, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"pipeline": pipeline, "protocol": protocol, "seed": seed, "config": cfg,
                  "manifest_sha256": fingerprint(), "state_dict": best_state, "mean": mean, "std": std,
                  "best_epoch": best_epoch, "training_examples": len(train_ids), "validation_examples": len(val_ids),
                  "training_seconds": time.perf_counter() - started, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                  "selection_metric": cfg["checkpoint_selection"], "history": history, "max_epochs": maximum}
    if pipeline in ("fusion", "speech_control"):
        speech_path = checkpoint_path(protocol, seed, "speech", tag)
        checkpoint["speech_checkpoint_sha256"] = hashlib.sha256(speech_path.read_bytes()).hexdigest()
    torch.save(checkpoint, path)
    directory = ROOT / "Results" / "training" / (protocol + tag) / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pipeline}.json").write_text(json.dumps({k: v for k, v in checkpoint.items() if k not in ("state_dict", "mean", "std")}, indent=2) + "\n")
    print(f"Saved {path.relative_to(ROOT)}; best epoch {best_epoch}; {checkpoint['training_seconds']:.1f}s", flush=True)


def evaluate(pipeline, protocol, seed, tag=""):
    seed_everything(seed)
    rows, data = manifest(), load_features()
    ids = indices(rows, protocol, "test")
    ckpt = load_checkpoint(checkpoint_path(protocol, seed, pipeline, tag))
    y = np.array([data["labels"][i] for i in ids])
    embeddings, logits = [], []
    with torch.inference_mode():
        if pipeline == "speech":
            model = SpeechModel().eval()
            model.load_state_dict(ckpt["state_dict"])
            loader = DataLoader(AudioDataset(data, ids, ckpt["mean"], ckpt["std"]), batch_size=config()["batch_size"], collate_fn=collate_audio)
            for x, lengths, _, _ in loader:
                e = model.encoder(x, lengths)
                embeddings.append(e)
                logits.append(model.head(e))
            logits = torch.cat(logits)
            embeddings = torch.cat(embeddings)
        else:
            if pipeline in ("fusion", "speech_control"):
                current_sha = hashlib.sha256(checkpoint_path(protocol, seed, "speech", tag).read_bytes()).hexdigest()
                assert ckpt["speech_checkpoint_sha256"] == current_sha, "Speech encoder changed after head training."
            model = Head(*head_dimensions(pipeline)).eval()
            model.load_state_dict(ckpt["state_dict"])
            raw = raw_features(data, pipeline, protocol, seed, tag)[ids]
            x = (raw - ckpt["mean"]) / ckpt["std"]
            logits, embeddings = model(x, return_embedding=True)
            if pipeline == "text":
                embeddings = raw  # Required contextual block representation, before classifier.
    probabilities = logits.softmax(1).numpy()
    predicted = probabilities.argmax(1)
    result = {"pipeline": pipeline, "protocol": protocol, "seed": seed, "n_test": len(ids),
              "best_epoch": ckpt["best_epoch"], "trainable_parameters": ckpt["trainable_parameters"],
              "training_seconds": ckpt["training_seconds"], **metrics(y, predicted),
              "confusion_matrix": confusion_matrix(y, predicted, labels=list(range(7))).tolist(),
              "per_class": classification_report(y, predicted, labels=list(range(7)), target_names=LABELS, output_dict=True, zero_division=0)}
    if pipeline == "fusion":
        with torch.inference_mode():
            shuffled = raw.clone()
            shuffled[:, 96:] = raw[torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)), 96:]
            zeroed = raw.clone()
            zeroed[:, 96:] = ckpt["mean"][96:]
            result["diagnostics"] = {"shuffled_text": metrics(y, model((shuffled - ckpt["mean"]) / ckpt["std"]).argmax(1).numpy()),
                                     "mean_text": metrics(y, model((zeroed - ckpt["mean"]) / ckpt["std"]).argmax(1).numpy())}
    directory = ROOT / "Results" / "evaluation" / (protocol + tag) / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pipeline}.json").write_text(json.dumps(result, indent=2) + "\n")
    import csv
    predpath = ROOT / "Results/predictions" / f"{protocol}{tag}-{seed}-{pipeline}.csv"
    with predpath.open("w", newline="") as f:
        fields = ["id", "speaker", "word", "transcript", "true_label", "predicted_label", "confidence"] + [f"p_{e}" for e in LABELS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j, i in enumerate(ids):
            writer.writerow({**{k: rows[i][k] for k in ("id", "speaker", "word", "transcript")},
                             "true_label": LABELS[y[j]], "predicted_label": LABELS[predicted[j]], "confidence": probabilities[j].max(),
                             **dict(zip([f"p_{e}" for e in LABELS], probabilities[j]))})
    np.savez_compressed(directory / f"{pipeline}-embeddings.npz", x=embeddings.numpy(), y=y,
                        ids=np.array([rows[i]["id"] for i in ids]), speakers=np.array([rows[i]["speaker"] for i in ids]))
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_class", "confusion_matrix")}), flush=True)


def main(pipeline=None, action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if pipeline is None:
        parser.add_argument("--pipeline", choices=["speech", "text", "fusion", "speech_control"], required=True)
    if action is None:
        parser.add_argument("--action", choices=["train", "test"], required=True)
    parser.add_argument("--protocol", choices=["main", "oaf_to_yaf", "yaf_to_oaf"], default="main")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    selected_pipeline, selected_action = pipeline or args.pipeline, action or args.action
    if selected_action == "train":
        train(selected_pipeline, args.protocol, args.seed, args.max_epochs, args.tag)
    else:
        evaluate(selected_pipeline, args.protocol, args.seed, args.tag)


if __name__ == "__main__":
    main()
