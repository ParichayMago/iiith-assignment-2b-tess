"""Verify acoustic units, pairing, train-only references and provenance safeguards."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from analyze_acoustics import measure, paired_differences, reference_distances, DISTANCE_FEATURES
from tess_emotion.provenance import same_execution_config, verify_training_snapshot


def test_acoustic_units_and_amplitude_invariance():
    rate = 16000
    audio = .2 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)
    result, half = measure(audio), measure(audio / 2)
    assert result['f0_median_hz'] == pytest.approx(200, abs=.5)
    assert result['rms_dbfs'] == pytest.approx(20 * np.log10(.2 / np.sqrt(2)), abs=.01)
    assert result['centroid_hz'] == pytest.approx(200, abs=10)
    assert result['f0_range_st'] < .05
    assert result['voiced_fraction'] > .95
    assert result['duration_s'] == 1
    assert half['rms_dbfs'] - result['rms_dbfs'] == pytest.approx(-6.0206, abs=.001)
    assert half['f0_median_hz'] == pytest.approx(result['f0_median_hz'], abs=.01)


def test_pairing_is_by_word_not_row_order_and_rejects_missing_pairs():
    frame = pd.DataFrame({'word': ['b', 'a', 'a', 'b'], 'speaker': ['YAF', 'OAF', 'YAF', 'OAF'], 'f0_median_hz': [400., 100., 200., 200.]})
    np.testing.assert_allclose(paired_differences(frame, 'f0_median_hz'), [12., 12.])
    with pytest.raises(AssertionError):
        paired_differences(frame.iloc[:-1], 'f0_median_hz')


def test_test_case_never_enters_reference_scales_or_centres():
    train = pd.DataFrame({name: [1., 2., 8., 9.] for name in DISTANCE_FEATURES})
    train['emotion'] = ['a', 'a', 'b', 'b']
    case = pd.Series({name: 2. for name in DISTANCE_FEATURES})
    scale, centre, distances = reference_distances(train, case)
    other_scale, other_centre, _ = reference_distances(train, case * 1e6)
    pd.testing.assert_series_equal(scale, other_scale)
    pd.testing.assert_frame_equal(centre, other_centre)
    assert distances['a'] < distances['b']
    assert scale.iloc[0] == 6.5


def test_only_control_prose_is_ignored_for_checkpoint_resume():
    original = json.loads((ROOT / 'Results/training_config_original.json').read_text())
    corrected = json.loads((ROOT / 'configs/experiment.json').read_text())
    assert same_execution_config(original, corrected)
    changed = {**corrected, 'learning_rate': corrected['learning_rate'] * 10}
    assert not same_execution_config(original, changed)
    assert not same_execution_config(original, {**corrected, 'new_setting': 1})


def test_provenance_accepts_exact_amendment_and_rejects_other_changes(tmp_path):
    snapshot = json.loads((ROOT / 'Results/run_manifest.json').read_text())
    paths = [*snapshot['source_sha256'], 'Results/training_config_original.json', 'Results/config_metadata_amendment.json']
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert verify_training_snapshot(tmp_path, snapshot) == ['configs/experiment.json: control description only']
    config_path = tmp_path / 'configs/experiment.json'
    config_path.write_text(config_path.read_text().replace('0.001,', '0.002,'))
    with pytest.raises(AssertionError):
        verify_training_snapshot(tmp_path, snapshot)
    shutil.copyfile(ROOT / 'configs/experiment.json', config_path)
    (tmp_path / 'Results/training_config_original.json').write_text('{}')
    with pytest.raises(AssertionError):
        verify_training_snapshot(tmp_path, snapshot)
