"""Validate the documented prose-only correction without rewriting training history."""
import hashlib
import json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_execution_config(first, second):
    # This field describes the control; head_dimensions() defines its architecture.
    return {k: v for k, v in first.items() if k != 'control'} == {
        k: v for k, v in second.items() if k != 'control'}


def verify_training_snapshot(root, snapshot):
    amendments = []
    for name, expected in snapshot['source_sha256'].items():
        if sha256(root / name) == expected:
            continue
        assert name == 'configs/experiment.json', f'Training source changed: {name}'
        amendment = json.loads((root / 'Results/config_metadata_amendment.json').read_text())
        original = root / amendment['original_snapshot']
        assert sha256(original) == expected == amendment['original_sha256']
        assert sha256(root / name) == amendment['corrected_sha256']
        before, after = json.loads(original.read_text()), json.loads((root / name).read_text())
        assert set(before) == set(after)
        assert {k for k in before if before[k] != after[k]} == {'control'}
        assert same_execution_config(before, after)
        assert amendment['field'] == 'control'
        assert before['control'] == amendment['before'] and after['control'] == amendment['after']
        amendments.append(name + ': control description only')
    return amendments
