# Assignment 2b: multimodal emotion recognition on TESS

Parichay Mago · IIIT Hyderabad Speech Processing Lab assignment

Three required systems predict seven acted emotions from speech, reconstructed text,
or both. A fourth speech-only control has approximately the same classifier parameter
count as fusion. The implementation follows the supplied preprocessing, feature
extraction, temporal/contextual modeling, fusion and classification blocks.

## Dataset audit comes first

Use the [Kaggle dataset named in the assignment](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
The original corpus is [TESS, Pichora-Fuller and Dupuis (2020), DOI 10.5683/SP2/E8H2MF](https://doi.org/10.5683/SP2/E8H2MF).

The downloaded archive contains **5,600 WAV entries but only 2,800 unique recordings**.
Two directory trees are byte-identical copies. `scripts/audit_dataset.py` verifies
their hashes, retains one copy, and checks decoded-audio hashes across distinct IDs.
It records the `OA_bite_neutral.wav` speaker-prefix typo and resolves it using its
`OAF_neutral` parent directory. It also finds one 96 kHz file (`OAF_food_fear`) and one
clipped file (`OAF_pad_ps`); all other files have sample rate 24,414 Hz. All files are
retained, and sample rates are handled explicitly during preprocessing.

No transcript files are present in this archive. Text is reconstructed as
`say the word <target>` from the corpus's documented carrier phrase and the target-word
field of the filename. **Emotion suffixes, speaker identifiers, paths and recording
IDs never enter the networks.** A local Whisper tiny.en diagnostic matched the
carrier and target on 16 preselected examples, including both audio anomalies; this
is an ASR spot-check, not a full human transcription audit. ASR predictions never
become training or test inputs.

Every unique transcript appears twice in each of seven classes. Therefore, any
deterministic classifier receiving only these transcripts has exactly **1/7 test
accuracy** on each balanced partition. A stronger text model cannot recover emotion
information absent from its input. Fusion gains must be measured rather than assumed.

See `Results/dataset_audit.json`, `Results/transcript_spotcheck.json`, and the manifest
and duplicate inventory under `data/manifests/` for machine-readable evidence.

## Reproduce

Tested on macOS ARM64, Python 3.12, with CPU training and four PyTorch threads. The
frozen model revision and every experiment setting are in `configs/experiment.json`.
`requirements.txt` pins training dependencies; `requirements-analysis.txt` includes
those pins plus Praat/Parselmouth for the post-hoc acoustic analysis. Internet access is needed
initially for the dataset and pretrained model; the experiments use no hosted API.

```bash
git clone https://github.com/ParichayMago/iiith-speech-assignment-2.git
cd iiith-speech-assignment-2/assignment-2b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
mkdir -p .local/logs
curl -fL --retry 2 \
  https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess \
  -o .local/tess.zip
python scripts/audit_dataset.py .local/tess.zip
python scripts/prepare_features.py
python -m pytest -q tests
python scripts/run_experiments.py --stage train
python scripts/run_experiments.py --stage test
python scripts/analyze_results.py
python scripts/analyze_acoustics.py
python scripts/build_report.py
python scripts/verify_results.py
```

If Kaggle requires authentication, download the archive from the dataset page and
save it as `.local/tess.zip`. Its SHA-256 is recorded in `Results/dataset_audit.json`.
The optional transcript audit uses FFmpeg (install it separately), then:

```bash
python scripts/check_transcripts.py
```

Individual entry points required by the brief:

```bash
python models/speech_pipeline/train.py --protocol main --seed 13
python models/speech_pipeline/test.py --protocol main --seed 13
python models/text_pipeline/train.py --protocol main --seed 13
python models/text_pipeline/test.py --protocol main --seed 13
python models/fusion_pipeline/train.py --protocol main --seed 13
python models/fusion_pipeline/test.py --protocol main --seed 13
```

Fusion requires the speech checkpoint for the same protocol and seed. A single run
can be limited to one epoch with `--max-epochs 1 --tag=-smoke`; smoke checkpoints have
a separate location and are excluded from the reported experiment tables.

`run_experiments.py` resumes existing checkpoints. To retrain after changing the
configuration or implementation, remove/move the affected checkpoints and rerun;
do not combine checkpoints from different experiments. Saved checkpoints contain
normalization statistics, hyperparameters, training history and the manifest hash.
Evaluation refuses stale dataset hashes and mismatched fusion/speech checkpoints.
The descriptive `control` field was corrected after training. Its original bytes and
the exact prose-only amendment are preserved in `Results/training_config_original.json`
and `Results/config_metadata_amendment.json`. No execution setting or trained weight
changed. Resume ignores only that descriptive field; result verification validates
the original source hashes and permits only this exact documented amendment.

## Fixed protocols

All systems use the same 140/30/30 target-word partition, split seed 1701. A word's
recordings stay together. Split checks reject overlapping words, IDs or decoded
audio hashes and verify class balance.

| Protocol | Training | Validation | Test | What it measures |
|---|---:|---:|---:|---|
| `main` | 1,960, both speakers | 420, both speakers | 420, both speakers | Unseen target words from known speakers |
| `oaf_to_yaf` | 980, OAF | 210, OAF | 210, YAF | Transfer from OAF to YAF on unseen words |
| `yaf_to_oaf` | 980, YAF | 210, YAF | 210, OAF | Transfer from YAF to OAF on unseen words |

In each transfer experiment, 1,400 other recordings are explicitly excluded.
The two-speaker design cannot estimate broad population-level speaker generalization.
Three seeds (13, 37, 73) are run for every system and protocol. Model selection uses
validation macro-F1 with validation cross-entropy breaking ties. Test labels do not
select hyperparameters, checkpoints, epochs or seeds.

## Architectures and training

- **Preprocessing:** anti-aliased resampling to mono 16 kHz; 20 ms RMS frames identify
  leading/trailing regions below 35 dB relative to the maximum frame RMS, with a
  50 ms preservation margin. No per-clip amplitude normalization is applied.
- **Speech features:** 64 HTK mel bands from a 512-point STFT, 25 ms Hann window and
  10 ms hop, log-power floor 1e-8. Per-band means and standard deviations use training
  frames only. Variable-length batches have explicit masks.
- **Speech temporal model:** two 64-channel temporal convolutions (kernels 5 and 3,
  stride 2 each), per-frame LayerNorm, ReLU/dropout, bidirectional GRU with 48 units
  per direction, and masked attention pooling. Packed sequences prevent backward
  recurrence through padding. The 96-dimensional representation feeds a 64-unit MLP.
- **Text contextual model:** frozen `distilbert/distilbert-base-uncased`, revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`. WordPiece tokenization and transformer
  contextual states are followed by masked mean pooling excluding special tokens.
  The 768-dimensional representation feeds a 64-unit MLP. Unique transcripts are
  encoded once; trainable components receive training-partition labels only.
- **Fusion:** freeze the selected speech encoder and the text encoder; concatenate
  96 speech and 768 text features, standardize each dimension using training data,
  and train an 864→64→7 MLP. This prevents extra acoustic fine-tuning from being
  mistaken for an improvement caused by text.
- **Capacity control:** frozen speech encoder plus a 96→537→7 MLP (55,855 head
  parameters versus fusion's 55,815; difference below 0.1%). It has comparable
  parameter count and depth, although the architectures are not identical.
- **Optimization:** AdamW, learning rate 0.001, weight decay 1e-4, cross-entropy,
  gradient norm limit 5, batch size 64. At most 45 speech epochs / 60 head epochs;
  early stopping patience 8. Configurations were fixed before final test evaluation.

DistilBERT is pretrained externally; TESS is the only labeled emotion dataset used.
The optional Whisper audit is not a feature extractor for the emotion models.

## Results and analysis outputs

- `Results/accuracy_table.csv`: three-seed accuracy, macro-F1 and balanced accuracy.
- `Results/all_runs.csv`: every run, selected epoch, parameter count and elapsed time.
- `Results/per_emotion.csv` and `Results/plots/confusion_matrices.*`: class-level errors.
- `Results/fusion_comparison.csv`: paired corrections and newly introduced mistakes.
- `Results/fusion_diagnostics.csv`: shuffled-text and mean-text substitution diagnostics.
  These are input perturbations, not additional independently trained models.
- `Results/error_recurrence*.csv`: unique erroneous clips and errors recurring in all
  three seeds. `Results/fusion_change_events.csv` identifies every fusion correction
  and new mistake; `fusion_changes_by_emotion.csv` groups these events by true class.
- `Results/acoustic_features.csv`: measured voiced F0 median/spread, voiced fraction,
  RMS recorded level, spectral centroid and trimmed duration for all 2,800 recordings.
  `acoustic_test_summary.csv` gives held-out speaker/class medians and quartiles;
  `acoustic_speaker_shifts.csv` matches the same 30 test words between actors.
  `failure_acoustic_evidence.json` compares the five existing errors with training-only
  reference distributions. Methods, range-sensitivity checks and input/output hashes
  are in `acoustic_analysis.json`. This analysis does not tune or retrain any model.
- `Results/plots/representations.*`: PCA of temporal speech, contextual text and fused
  representations. Each PCA is fitted on training embeddings; identical test samples
  are plotted by emotion and speaker. Seed 13 was fixed in advance. PCA axes differ
  between modalities and two-dimensional overlap is not a classification metric.
- `Results/failure_cases.json` and failure waveforms: five actual errors, with paired
  predictions, five nearest training speech embeddings, and cautious interpretations.
  Neighbours use Euclidean distance after training-only feature standardization;
  they are descriptive, not an additional trained classifier. No human perceptual
  relabeling is claimed.
- `Results/evaluation/`, `Results/predictions/`, `Results/training/`: raw metrics,
  probabilities, embeddings and histories supporting every table and figure.
- `report.pdf`: architecture decisions, experiments, analysis and limitations.

Standard deviations describe training-seed variation on the same test set; they are
not confidence intervals over a population. Timing measurements are from a shared
local CPU and should not be interpreted as a controlled runtime benchmark.

`Results/verification.json` independently reconciles all 36 exported prediction
files against their metrics and summaries. `Results/fresh_reproduction.json` records
a fresh-source replay of all four main-protocol systems at seed 13: the rebuilt
manifest, selected weights, normalization statistics and prediction files matched
exactly. That replay regenerated task features and checkpoints using the same
installed dependency environment and cached pretrained-model download.
It predates the documentation correction and post-hoc acoustic analysis; their
additional checks do not constitute a new training replay.

## Acoustic analysis scope

`analyze_acoustics.py` uses the same resampling and saved trim boundaries as the
classifier, with all original WAV hashes verified. Parselmouth 0.4.7 / Praat 6.1.38
raw autocorrelation estimates F0 at 10 ms steps, primarily over 75-600 Hz, repeated
at 60-700 Hz as a range-sensitivity check. Recorded RMS dBFS is not calibrated
loudness; spectral centroids include voiced and unvoiced regions. Pitch estimates
can contain octave errors and have not been manually annotated.

Speaker shifts pair the same target words within each emotion, reporting median
YAF-minus-OAF differences and 2,000 word-pair bootstrap percentile intervals
(seed 1701). These intervals describe these two actors on these words, not a
population of speakers. Speaker, acting style and recording conditions are
confounded. Four fixed acoustic descriptors are compared with training-only class
medians using training IQR scales; proximity is descriptive and cannot establish
why a neural network erred. Case selection preceded this analysis and is error-focused.

## Repository and data access

Source, manifests, report and measured results are published in the
[public assignment repository](https://github.com/ParichayMago/iiith-speech-assignment-2).
Raw audio, cached features, environments and temporary files are ignored.
The dataset provider's metadata lists noncommercial/no-derivatives terms; use the
original download and consult the provider's current terms rather than redistributing
audio here. Checkpoints are reproducible with the commands above and available in
the [assignment release](https://github.com/ParichayMago/iiith-speech-assignment-2/releases/tag/assignment-2b).
The 7.7 MiB ZIP contains all 36 task-model checkpoints and SHA-256 checksums; extract
it into this `assignment-2b/` directory. The same checksum inventory is committed as
`Results/checkpoint_sha256.json`. Frozen DistilBERT weights are downloaded separately
at the pinned revision when generating features.

## References

1. Pichora-Fuller, M. K.; Dupuis, K. (2020). *Toronto emotional speech set (TESS).* Borealis, V1. https://doi.org/10.5683/SP2/E8H2MF
2. Assignment-specified Kaggle mirror: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
3. Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* https://arxiv.org/abs/1910.01108
4. Frozen model and revision: https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be
5. Praat raw autocorrelation method: https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_raw_autocorrelation.html
6. Parselmouth interface: https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_ac
