"""Read the TextGrids in Praat, verify coverage, and plot a sample alignment."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import parselmouth
from parselmouth.praat import call
from prepare_corpus import SPEECH_END

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def read_tiers(path: Path) -> dict[str, list[Interval]]:
    grid = call("Read from file", str(path))
    tiers = {}
    for tier in range(1, call(grid, "Get number of tiers") + 1):
        name = call(grid, "Get tier name", tier)
        intervals = []
        for number in range(1, call(grid, "Get number of intervals", tier) + 1):
            intervals.append(Interval(
                call(grid, "Get start time of interval", tier, number),
                call(grid, "Get end time of interval", tier, number),
                call(grid, "Get label of interval", tier, number),
            ))
        tiers[name] = intervals
    if set(tiers) != {"words", "phones"}:
        raise ValueError(f"{path.name} has unexpected tiers: {list(tiers)}")
    return tiers


def check_tiers(tiers: dict[str, list[Interval]], duration: float, stem: str) -> None:
    tolerance = 0.0001
    for name, intervals in tiers.items():
        previous_end = 0.0
        for item in intervals:
            if abs(previous_end - item.start) > tolerance or item.start >= item.end:
                raise ValueError(f"{stem}: invalid or discontinuous {name} tier near {item.start:.3f} s")
            previous_end = item.end
        if abs(previous_end - duration) > tolerance:
            raise ValueError(f"{stem}: {name} tier length differs from WAV duration")


def plot_sample(sound: parselmouth.Sound, tiers: dict[str, list[Interval]], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 4.8),
                             gridspec_kw={"height_ratios": [1.4, 2, 0.54, 0.54]})
    wave, spectrogram, words, phones = axes
    wave.plot(sound.xs(), sound.values[0], color="#155678", linewidth=0.5)
    wave.set_ylabel("Amplitude")
    wave.set_ylim(-1.05 * np.abs(sound.values).max(), 1.05 * np.abs(sound.values).max())
    spec = sound.to_spectrogram(window_length=0.008, maximum_frequency=6500, time_step=0.002)
    power = 10 * np.log10(np.maximum(spec.values, 1e-20))
    ceiling = np.percentile(power, 99.5)
    spectrogram.pcolormesh(spec.x_grid(), spec.y_grid() / 1000, power, cmap="Greys",
                           vmin=ceiling - 55, vmax=ceiling, shading="flat", rasterized=True)
    spectrogram.set_ylabel("Frequency (kHz)")
    for axis, tier, fill in [(words, "words", "#dcecf5"), (phones, "phones", "#f1e8d2")]:
        for interval in tiers[tier]:
            axis.axvspan(interval.start, interval.end, facecolor=fill if interval.label else "#f5f6f7",
                         edgecolor="#6d7c86", linewidth=0.45)
            width = interval.end - interval.start
            axis.text(interval.start + width / 2, 0.5, interval.label, ha="center", va="center",
                      fontsize=8 if width > 0.075 else 6, clip_on=True)
        axis.set_yticks([])
        axis.set_ylabel(tier.title(), rotation=0, va="center", ha="right")
    for item in tiers["words"]:
        if item.label:
            for axis in (wave, spectrogram):
                for boundary in (item.start, item.end):
                    axis.axvline(boundary, color="#007aa4", alpha=0.65, linewidth=0.55)
    phones.set_xlabel("Time (s)")
    phones.set_xlim(0, sound.duration)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", bottom=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.11, hspace=0.09)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def inspect(corpus: Path, alignments: Path, results: Path, sample: str,
            baseline: Path | None) -> None:
    expected = {lab.relative_to(corpus).with_suffix(".TextGrid") for lab in corpus.rglob("*.lab")}
    seen = {grid.relative_to(alignments) for grid in alignments.rglob("*.TextGrid")}
    if len(expected) != 6 or seen != expected:
        raise ValueError(f"TextGrid coverage mismatch: missing {expected - seen}, extra {seen - expected}")

    results.mkdir(parents=True, exist_ok=True)
    rows = []
    boundary_review = []
    for relative_path in sorted(expected):
        sound = parselmouth.Sound(str(corpus / relative_path.with_suffix(".wav")))
        lab = (corpus / relative_path.with_suffix(".lab")).read_text(encoding="utf-8").split()
        tiers = read_tiers(alignments / relative_path)
        check_tiers(tiers, sound.duration, relative_path.stem)
        words = [item for item in tiers["words"] if item.label]
        phones = [item for item in tiers["phones"] if item.label and item.label != "sil"]
        if [word.label for word in words] != lab:
            raise ValueError(f"{relative_path.stem}: aligned words differ from prepared transcript")
        if any(phone.label in {"spn", "<unk>"} for phone in phones):
            raise ValueError(f"{relative_path.stem}: unresolved word or unknown phone")
        rows.append({
            "utterance": relative_path.stem, "duration_s": f"{sound.duration:.3f}",
            "words": len(words), "phones": len(phones),
            "first_word_start_s": f"{words[0].start:.3f}", "last_word_end_s": f"{words[-1].end:.3f}",
            "phones_at_most_20_ms": sum(phone.end - phone.start <= 0.02001 for phone in phones),
        })
        if baseline is not None and relative_path.stem in SPEECH_END:
            original = read_tiers(baseline / relative_path)
            original_phone = [item for item in original["phones"] if item.label][-1]
            if original_phone.label != phones[-1].label or phones[-1].end >= original_phone.end:
                raise ValueError(f"{relative_path.stem}: expected a shorter final phone after segmentation")
            boundary_review.append({
                "utterance": relative_path.stem, "last_word": words[-1].label, "phone": phones[-1].label,
                "baseline_start_s": f"{original_phone.start:.3f}", "baseline_end_s": f"{original_phone.end:.3f}",
                "reviewed_start_s": f"{phones[-1].start:.3f}", "reviewed_end_s": f"{phones[-1].end:.3f}",
            })
        if relative_path.stem == sample:
            plot_sample(sound, tiers, results / "sample_alignment.png")

    with (results / "alignment_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if boundary_review:
        with (results / "boundary_review.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=boundary_review[0].keys())
            writer.writeheader()
            writer.writerows(boundary_review)
    print(f"Praat {parselmouth.PRAAT_VERSION}: checked {len(rows)} TextGrids and "
          f"{sum(row['words'] for row in rows)} aligned words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("alignments", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--sample", default="ISLE_SESS0131_BLOCKD02_03_sprt1")
    parser.add_argument("--baseline", type=Path, help="Compare the last phones to the full-clip alignment")
    args = parser.parse_args()
    inspect(args.corpus, args.alignments, args.results, args.sample, args.baseline)
