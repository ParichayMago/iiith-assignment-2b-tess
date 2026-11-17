"""Checks for leakage, feature normalization and variable-length sequence handling."""
import importlib.util
from pathlib import Path
import sys
from collections import Counter, defaultdict
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_dataset import validate_splits
from tess_emotion.data import manifest, audio_stats
from tess_emotion.networks import SpeechEncoder, Head


def test_real_partitions_are_balanced_and_disjoint():
    validate_splits(manifest())


def test_leakage_is_detected():
    rows = manifest()
    train = next(r for r in rows if r['split_main'] == 'train')
    test = next(r for r in rows if r['split_main'] == 'test')
    test['pcm_sha256'] = train['pcm_sha256']
    with pytest.raises(AssertionError):
        validate_splits(rows)


def test_every_transcript_has_all_labels_in_equal_numbers():
    counts = defaultdict(Counter)
    for row in manifest():
        counts[row['transcript']][row['emotion']] += 1
        assert row['transcript'] == 'say the word ' + row['word']
    assert len(counts) == 200
    assert all(len(c) == 7 and set(c.values()) == {2} for c in counts.values())


def test_normalization_ignores_held_out_features():
    audio = [torch.tensor([[1., 2.], [3., 4.]]), torch.full((3, 2), 1e9)]
    mean, std = audio_stats(audio, [0])
    torch.testing.assert_close(mean, torch.tensor([2., 3.]))
    audio[1].fill_(-1e9)
    other_mean, other_std = audio_stats(audio, [0])
    torch.testing.assert_close(mean, other_mean)
    torch.testing.assert_close(std, other_std)


def test_padding_does_not_change_speech_representation():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = SpeechEncoder().eval()
    x = torch.randn(1, 71, 64)
    padded = torch.cat([x, torch.zeros(1, 29, 64)], dim=1)
    with torch.inference_mode():
        alone = model(x, torch.tensor([71]))
        extra_padding = model(padded, torch.tensor([71]))
    torch.testing.assert_close(alone, extra_padding, atol=1e-6, rtol=1e-5)


def test_control_and_fusion_have_comparable_parameter_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())
    fusion, control = count(Head(864)), count(Head(96, 537))
    assert abs(fusion - control) / fusion < .001
