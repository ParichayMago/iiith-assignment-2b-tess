# IIIT Hyderabad Assignment 2

Parichay Mago's work for the Speech Processing Lab assignment, covering both parts.

| Part | Implementation and instructions | Report | Outputs |
|---|---|---|---|
| 2a — MFA forced alignment | [Setup and reproduction](assignment-2a/README.md) | [Short PDF](assignment-2a/report.pdf) | [Six word/phone TextGrids](assignment-2a/results/textgrids/) |
| 2b — Multimodal emotion recognition | [Setup and reproduction](assignment-2b/README.md) | [Full PDF](assignment-2b/report.pdf) | [Accuracy tables and plots](assignment-2b/Results/) |

[The combined PDF](report.pdf) contains both reports for the Google Form's report upload.
It preserves the standalone reports and adds bookmarks for each part. To rebuild it
after editing either report, run `assignment-2b/.venv/bin/python scripts/build_submission_report.py`
from the repository root, using the dependency environment described in 2b's README.

## Assignment 2a

The six supplied audio/transcript pairs are aligned with Montreal Forced Aligner.
The reproducible pipeline prepares the corpus, checks vocabulary coverage, applies
documented pronunciation additions, generates word and phone TextGrids, and checks
them using Praat through Parselmouth. The report shows the alignment visualization
and documents two trailing-silence errors and the speech windows used to review them.

## Assignment 2b

Reproducible speech-only, text-only and multimodal emotion recognition on TESS.

- [Implementation and reproduction instructions](assignment-2b/README.md)
- [Report with measured acoustic analysis](assignment-2b/report.pdf)
- [Dataset audit](assignment-2b/Results/dataset_audit.json)
- [Measured three-seed results](assignment-2b/Results/accuracy_table.csv)
- [Independent result verification](assignment-2b/Results/verification.json)
- [Acoustic measurements and analysis provenance](assignment-2b/Results/acoustic_analysis.json)
- [Fresh-source reproduction check](assignment-2b/Results/fresh_reproduction.json)
- [All 36 trained checkpoints](https://github.com/ParichayMago/iiith-speech-assignment-2/releases/tag/assignment-2b)

The audit removes 2,800 duplicate archive entries before splitting, leaving the
complete 2,800-recording corpus. It verifies word/audio separation and reconstructs
text without emotion labels. All three seeds and both speaker-transfer directions
are reported, including the weaker transfer results.

| Evaluation | Speech accuracy | Text accuracy | Fusion accuracy |
|---|---:|---:|---:|
| Unseen words, known speakers | 99.44% | 14.29% | 99.52% |
| OAF training to YAF test | 29.05% | 14.29% | 28.73% |
| YAF training to OAF test | 35.08% | 14.29% | 35.24% |

Means across three seeds; standard deviations, macro-F1 and the capacity control
are included in the report. Near-perfect known-speaker scores do not transfer to
unseen speakers. Clean TESS transcripts repeat across every emotion, so text-only
accuracy is necessarily 1/7 on these balanced splits. Fusion does not demonstrate
a consistent advantage over speech and the parameter-matched classifier control.

Author: Parichay Mago. Dataset and pretrained-model attribution are in the project
README. Raw audio is downloaded from the provider and is not included here.

## Submission instructions

The lab's email asks for both parts in a public GitHub repository and submission
through [the Google Form](https://forms.gle/hCyXpEyks9NAnGPS9). The form requires a
name, email address, repository link and a PDF upload. The 2a brief additionally
asks for the short 2a PDF by email. Any linked files must be publicly accessible.
The stated deadline is Sunday, November 22, 2026 at 11:59 p.m.
