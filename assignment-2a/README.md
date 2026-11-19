# Assignment 2a · Forced alignment with MFA

This project aligns the six English audio–transcript pairs supplied by the IIIT Hyderabad Speech Processing Lab. Each output TextGrid contains `words` and `phones` tiers on the original recording's time axis.

The public repository contains both assignment parts. From the repository root, enter `assignment-2a/` before running the commands below.

## Reproduce the alignment

Install [Miniforge](https://conda-forge.org/download/), FFmpeg, and [uv](https://docs.astral.sh/uv/getting-started/installation/). Use the original `Assignment_2.zip` from the lab. The commands below were tested on Apple Silicon with MFA 3.3.10 and Python 3.12 for the preparation and inspection scripts.

```bash
conda create -n iiith-mfa -c conda-forge montreal-forced-aligner=3.3.10 kalpy=0.10.5
conda activate iiith-mfa
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-review.txt
REVIEW_PYTHON="$PWD/.venv/bin/python" bash run_2a.sh /path/to/Assignment_2.zip rerun
python scripts/build_report.py rerun rerun/report.pdf
```

Use a fresh directory for each run. The script downloads the original English MFA acoustic model and US English MFA dictionary release files, checks their SHA-256 hashes, and stages the run under a path without spaces for Kaldi. To reuse existing model files, pass their directory as the third argument. The two filenames are `english_mfa.zip` and `english_us_mfa.dict`.

## Corpus and alignment

- `scripts/prepare_corpus.py` pairs files by name and groups recordings by the repeated speaker identifier in their filenames. FFmpeg converts the three NIST SPHERE files and three RIFF WAV files to mono 16 kHz PCM WAV. The script checks that the decoded samples are identical to those of the supplied files. Transcript cleanup standardizes case, punctuation, and hyphenation and expands the letters in `S.J.C.'s` and `WBUR's` into individual tokens.
- The [English MFA acoustic model v3.1.0](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20MFA%20acoustic%20model%20v3_1_0.html) and [US English MFA dictionary v3.1.0](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v3_1_0.html) share the MFA English phone inventory. MFA's stock OOV file identifies six missing words after clitic handling. `lexicon_additions.dict` contains selected candidates from the [US English MFA G2P model v3.0.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/g2p-english_us_mfa-v3.0.0). A vocabulary check with the augmented dictionary finds zero unresolved tokens. The only `bed` occurs at the end of a phrase, so both reported runs use its dictionary variant ending in `d`.
- The full-clip run places the last phone of `bait` and `bed` across long quiet tails. I marked speech windows ending at 2.40 s and 2.20 s, respectively, after examining the waveform and spectrogram. MFA re-aligns those windows using input TextGrid tiers and keeps the original time axis. The final TextGrid endpoints are estimated by MFA within those windows.
- `scripts/inspect_alignments.py` uses [Praat via Parselmouth](https://parselmouth.readthedocs.io/en/latest/api/parselmouth.praat.html) to read all six output grids, verify the word labels against the prepared transcript, check each tier's coverage, and plot a sample waveform, spectrogram, and alignment. The supplied data has no reference phone boundaries for measuring timing error.

## Results and visual review

Use `results/textgrids/` for the six final alignments. `results/baseline/` contains the first pass over full clips for comparison. The CSVs record per-file counts and the two reviewed final-phone intervals. [The PDF report](report.pdf) includes the sample visualization and findings. Regenerate it with `python scripts/build_report.py results report.pdf`.

To inspect a pair interactively in [Praat](https://praat.org/), prepare the corpus with `python3 scripts/prepare_corpus.py /path/to/Assignment_2.zip review-corpus`. Open a WAV from that directory and its matching TextGrid from `results/textgrids/` with **Open → Read from file**. Select both objects and choose **View & Edit**.
