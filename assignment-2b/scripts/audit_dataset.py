"""Audit and deduplicate the assignment's Kaggle archive before making splits.

Never use directory paths, recording IDs, speaker IDs or emotion suffixes as
model inputs. Text is reconstructed from the documented carrier and target word.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = {"angry": "anger", "disgust": "disgust", "fear": "fear",
            "happy": "happiness", "neutral": "neutral", "ps": "pleasant_surprise", "sad": "sadness"}
LABELS = sorted(EMOTIONS.values())
PATTERN = re.compile(r"^(OAF|YAF)_([a-z]+)_(angry|disgust|fear|happy|neutral|ps|sad)\.wav$", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_words(words, seed=1701):
    words = sorted(words)
    random.Random(seed).shuffle(words)
    assert len(words) == 200, f"Expected 200 words, found {len(words)}"
    return {word: ("train" if i < 140 else "validation" if i < 170 else "test")
            for i, word in enumerate(words)}


def validate_splits(rows):
    for protocol in ("main", "oaf_to_yaf", "yaf_to_oaf"):
        by_split = {s: [r for r in rows if r[f"split_{protocol}"] == s]
                    for s in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for key in ("word", "pcm_sha256", "id"):
                assert not ({r[key] for r in by_split[a]} & {r[key] for r in by_split[b]}), (protocol, a, b, key)
        for split, part in by_split.items():
            counts = Counter(r["emotion"] for r in part)
            assert len(counts) == 7 and len(set(counts.values())) == 1, (protocol, split, counts)
        if protocol != "main":
            assert {r["speaker"] for r in by_split["train"]}.isdisjoint({r["speaker"] for r in by_split["test"]})


def audit(archive_path, seed=1701):
    audio_dir = ROOT / "data/audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    duplicates, rows, seen, pcm_seen, repairs = [], [], {}, {}, []
    with ZipFile(archive_path) as archive:
        names = sorted(archive.namelist(), key=lambda n: (len(n), n))
        for name in names:
            if not name.lower().endswith(".wav"):
                raise ValueError(f"Unexpected archive entry: {name}")
            filename = Path(name).name
            if filename == "OA_bite_neutral.wav" and Path(name).parent.name == "OAF_neutral":
                repairs.append({"archive_member": name, "canonical_filename": "OAF_bite_neutral.wav",
                                "reason": "Missing F in speaker prefix; enclosing OAF_neutral directory supplies speaker metadata."})
                filename = "OAF_bite_neutral.wav"
            match = PATTERN.fullmatch(filename)
            if not match:
                raise ValueError(f"Unexpected filename: {filename}")
            speaker, word, suffix = match.groups()
            speaker, word, suffix = speaker.upper(), word.lower(), suffix.lower()
            identity = f"{speaker}_{word}_{suffix}"
            content = archive.read(name)
            digest = sha(content)
            if identity in seen:
                assert seen[identity][1] == digest, f"Conflicting duplicate: {identity}"
                duplicates.append({"id": identity, "kept": seen[identity][0], "duplicate": name, "sha256": digest})
                continue
            seen[identity] = (name, digest)
            audio, rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
            assert np.isfinite(audio).all() and len(audio) > 0, identity
            assert audio.shape[1] == 1, (identity, audio.shape)
            pcm_digest = sha(audio.astype("<f4").tobytes())
            assert pcm_digest not in pcm_seen, f"Identical audio under distinct IDs: {identity}, {pcm_seen.get(pcm_digest)}"
            pcm_seen[pcm_digest] = identity
            path = audio_dir / f"{identity}.wav"
            if not path.exists() or sha(path.read_bytes()) != digest:
                path.write_bytes(content)
            rows.append({"id": identity, "audio_path": str(path.relative_to(ROOT)), "speaker": speaker,
                         "word": word, "transcript": f"say the word {word}", "emotion": EMOTIONS[suffix],
                         "label": LABELS.index(EMOTIONS[suffix]), "sample_rate": rate,
                         "samples": len(audio), "duration_s": len(audio) / rate, "channels": audio.shape[1],
                         "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio ** 2))),
                         "clipped_fraction": float(np.mean(np.abs(audio) >= .9999)),
                         "file_sha256": digest, "pcm_sha256": pcm_digest, "archive_member": name})
    assert len(rows) == 2800, len(rows)
    cells = Counter((r["speaker"], r["word"], r["emotion"]) for r in rows)
    words = {r["word"] for r in rows}
    missing = [(s, w, e) for s in ("OAF", "YAF") for w in words for e in LABELS if cells[s, w, e] != 1]
    assert not missing, missing[:30]
    splits = split_words(words, seed)
    for r in rows:
        r["split_main"] = splits[r["word"]]
        for protocol, train_speaker in (("oaf_to_yaf", "OAF"), ("yaf_to_oaf", "YAF")):
            s = r["split_main"]
            use = (s in ("train", "validation") and r["speaker"] == train_speaker) or (s == "test" and r["speaker"] != train_speaker)
            r[f"split_{protocol}"] = s if use else "excluded"
    rows.sort(key=lambda r: r["id"])
    validate_splits(rows)
    write_csv(ROOT / "data/manifests/manifest.csv", rows)
    write_csv(ROOT / "data/manifests/duplicate_entries.csv", duplicates)
    (ROOT / "data/manifests/word_splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n")
    durations = np.array([r["duration_s"] for r in rows])
    theoretical = {}
    for split in ("train", "validation", "test"):
        counts = defaultdict(Counter)
        for r in rows:
            if r["split_main"] == split:
                counts[r["transcript"]][r["emotion"]] += 1
        theoretical[split] = sum(max(v.values()) for v in counts.values()) / sum(sum(v.values()) for v in counts.values())
    report = {
        "source_url": "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess",
        "original_dataset_doi": "https://doi.org/10.5683/SP2/E8H2MF",
        "archive_sha256": sha(archive_path.read_bytes()), "archive_wav_entries": len(names),
        "unique_recordings": len(rows), "identical_duplicate_entries_removed": len(duplicates),
        "duplicate_audio_across_distinct_ids": 0, "missing_speaker_word_emotion_combinations": len(missing),
        "filename_repairs": repairs,
        "speakers": sorted({r["speaker"] for r in rows}), "target_words": len(words), "labels": LABELS,
        "class_counts": dict(Counter(r["emotion"] for r in rows)),
        "sample_rate_counts": dict(Counter(r["sample_rate"] for r in rows)),
        "duration_s": {"min": float(durations.min()), "median": float(np.median(durations)), "max": float(durations.max()), "total": float(durations.sum())},
        "files_with_clipping": sum(r["clipped_fraction"] > 0 for r in rows),
        "transcript_provenance": "No transcript files in the supplied Kaggle archive. Reconstructed from the original dataset's documented carrier phrase 'Say the word _____' and target-word filename field; not verbatim human annotations.",
        "text_only_max_deterministic_accuracy": theoretical,
        "split_seed": seed,
        "split_counts": {p: dict(Counter(r[f"split_{p}"] for r in rows)) for p in ("main", "oaf_to_yaf", "yaf_to_oaf")},
        "checks_passed": ["audio decodes", "finite mono samples", "complete 2x200x7 grid", "duplicate bytes match", "no duplicate PCM across distinct IDs", "no word/PCM/ID overlap between partitions", "balanced labels in every partition", "held-out speakers disjoint"],
        "license_note": "Kaggle lists CC BY-NC-ND 4.0; raw audio is excluded from the source repository. See the original provider's terms for reuse."
    }
    (ROOT / "Results/dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    audit(args.archive, args.seed)
