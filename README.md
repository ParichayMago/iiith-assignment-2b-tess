# IIIT Hyderabad Assignment 2b

Reproducible speech-only, text-only and multimodal emotion recognition on TESS.

- [Implementation and reproduction instructions](assignment-2b/README.md)
- [Seven-page report](assignment-2b/report.pdf)
- [Dataset audit](assignment-2b/Results/dataset_audit.json)
- [Measured three-seed results](assignment-2b/Results/accuracy_table.csv)
- [Independent result verification](assignment-2b/Results/verification.json)
- [Fresh-source reproduction check](assignment-2b/Results/fresh_reproduction.json)
- [All 36 trained checkpoints](https://github.com/ParichayMago/iiith-assignment-2b-tess/releases/tag/assignment-2b)

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
