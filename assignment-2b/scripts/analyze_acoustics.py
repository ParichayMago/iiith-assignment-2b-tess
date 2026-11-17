"""Post-hoc acoustic descriptions and paired error accounting; no model fitting."""
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
from scipy.signal import resample_poly
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tess_emotion.data import ROOT, LABELS, config

FEATURES = ['f0_median_hz', 'f0_range_st', 'rms_dbfs', 'centroid_hz', 'duration_s', 'voiced_fraction']
DISTANCE_FEATURES = ['f0_median_st', 'f0_range_st', 'rms_dbfs', 'centroid_hz']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (ROOT / 'Results' / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pitch_summary(sound, floor, ceiling):
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=floor, pitch_ceiling=ceiling,
                              max_number_of_candidates=15, very_accurate=False,
                              silence_threshold=.03, voicing_threshold=.45,
                              octave_cost=.01, octave_jump_cost=.35, voiced_unvoiced_cost=.14)
    values = pitch.selected_array['frequency']
    voiced = values[values > 0]
    if not len(voiced):
        return np.nan, np.nan, 0.
    return float(np.median(voiced)), float(12 * np.log2(np.percentile(voiced, 90) / np.percentile(voiced, 10))), float(np.mean(values > 0))


def measure(audio, rate=16000):
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=rate)
    f0, spread, fraction = pitch_summary(sound, 75, 600)
    alternative, _, _ = pitch_summary(sound, 60, 700)
    magnitude = torch.stft(torch.tensor(audio, dtype=torch.float32), n_fft=512, hop_length=160,
                           win_length=400, window=torch.hann_window(400), return_complex=True).abs().numpy()
    weights = magnitude.sum(0)
    active = weights > 1e-12
    centroid = (np.fft.rfftfreq(512, 1 / rate)[:, None] * magnitude).sum(0)[active] / weights[active]
    return {'f0_median_hz': f0, 'f0_median_st': 12 * np.log2(f0 / 100), 'f0_range_st': spread,
            'voiced_fraction': fraction, 'f0_alternative_hz': alternative,
            'rms_dbfs': float(20 * np.log10(max(np.sqrt(np.mean(audio.astype(np.float64) ** 2)), 1e-12))),
            'centroid_hz': float(centroid.mean()), 'duration_s': len(audio) / rate}


def paired_differences(frame, feature):
    pivot = frame.pivot(index='word', columns='speaker', values=feature).sort_index()
    assert set(pivot.columns) == {'OAF', 'YAF'} and pivot.notna().all().all()
    if feature in ['f0_median_hz', 'f0_alternative_hz']:
        return 12 * np.log2(pivot.YAF.to_numpy() / pivot.OAF.to_numpy())
    return pivot.YAF.to_numpy() - pivot.OAF.to_numpy()


def reference_distances(train, case):
    """Fixed four-descriptor RMS distance, scaled by pooled training IQR only."""
    scales = (train[DISTANCE_FEATURES].quantile(.75) - train[DISTANCE_FEATURES].quantile(.25)).clip(lower=1e-6)
    medians = train.groupby('emotion')[DISTANCE_FEATURES].median()
    distances = np.sqrt((((medians - case[DISTANCE_FEATURES].astype(float)) / scales) ** 2).mean(axis=1))
    return scales, medians, distances


def error_accounting(manifest, cfg):
    events, stability, summaries, predictions = [], [], [], []
    for protocol in cfg['protocols']:
        for seed in cfg['seeds']:
            pair = {}
            for pipeline in ['speech', 'fusion']:
                path = ROOT / f'Results/predictions/{protocol}-{seed}-{pipeline}.csv'
                frame = pd.read_csv(path).set_index('id')
                predictions.append(path)
                expected = set(manifest.loc[manifest['split_' + protocol] == 'test', 'id'])
                assert frame.index.is_unique and set(frame.index) == expected
                pair[pipeline] = frame
            speech, fusion = pair['speech'], pair['fusion'].loc[pair['speech'].index]
            assert (speech.true_label == fusion.true_label).all()
            for ident in speech.index:
                sw = speech.loc[ident, 'true_label'] != speech.loc[ident, 'predicted_label']
                fw = fusion.loc[ident, 'true_label'] != fusion.loc[ident, 'predicted_label']
                stability.append({'protocol': protocol, 'seed': seed, 'id': ident, 'emotion': speech.loc[ident, 'true_label'], 'speech_wrong': sw, 'fusion_wrong': fw})
                if sw != fw:
                    events.append({'protocol': protocol, 'seed': seed, 'id': ident, 'true_label': speech.loc[ident, 'true_label'], 'speech_prediction': speech.loc[ident, 'predicted_label'], 'fusion_prediction': fusion.loc[ident, 'predicted_label'], 'event': 'fix' if sw else 'break'})
    raw = pd.DataFrame(stability)
    grouped = raw.groupby(['protocol', 'id', 'emotion'])[['speech_wrong', 'fusion_wrong']].sum().reset_index()
    for protocol in cfg['protocols']:
        for pipeline in ['speech', 'fusion']:
            subset = grouped[grouped.protocol == protocol]
            column = pipeline + '_wrong'
            summaries.append({'protocol': protocol, 'pipeline': pipeline, 'test_clips': len(subset),
                              'error_events': int(subset[column].sum()), 'unique_wrong_clips': int((subset[column] > 0).sum()),
                              'wrong_all_seeds': int((subset[column] == len(cfg['seeds'])).sum())})
    grouped.to_csv(ROOT / 'Results/error_recurrence.csv', index=False)
    pd.DataFrame(summaries).to_csv(ROOT / 'Results/error_recurrence_summary.csv', index=False)
    event_frame = pd.DataFrame(events)
    event_frame.to_csv(ROOT / 'Results/fusion_change_events.csv', index=False)
    event_frame.groupby(['protocol', 'true_label', 'event']).size().unstack(fill_value=0).reindex(columns=['fix', 'break'], fill_value=0).to_csv(ROOT / 'Results/fusion_changes_by_emotion.csv')
    # Reconcile with the existing benchmark accounting, including zero-event seeds.
    prior = pd.read_csv(ROOT / 'Results/fusion_comparison.csv').query("reference == 'speech'")
    for row in prior.itertuples():
        subset = event_frame[(event_frame.protocol == row.protocol) & (event_frame.seed == row.seed)]
        assert (subset.event == 'fix').sum() == row.fusion_fixes
        assert (subset.event == 'break').sum() == row.fusion_breaks
    return predictions


def main():
    cfg = config()
    torch.set_num_threads(cfg['threads'])
    manifest = pd.read_csv(ROOT / 'data/manifests/manifest.csv')
    trims = {r['id']: r for r in json.loads((ROOT / 'Results/preprocessing.json').read_text())['recordings']}
    rows = []
    for i, row in enumerate(manifest.itertuples()):
        path = ROOT / row.audio_path
        assert digest(path) == row.file_sha256, f'Audio changed: {row.id}'
        audio, rate = sf.read(path, dtype='float32')
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        trim = trims[row.id]
        audio = audio[round(trim['start_s'] * 16000):round(trim['end_s'] * 16000)]
        rows.append({'id': row.id, **measure(audio)})
        if i % 500 == 0:
            print(f'Acoustic measurements {i}/{len(manifest)}', flush=True)
    features = pd.DataFrame(rows)
    assert features.id.is_unique and np.isfinite(features.drop(columns='id').to_numpy()).all()
    frame = manifest.rename(columns={'duration_s': 'original_duration_s'}).merge(features, on='id', validate='one_to_one')
    features.to_csv(ROOT / 'Results/acoustic_features.csv', index=False, float_format='%.10g')
    held = frame[frame.split_main == 'test']
    assert len(held) == 420
    stats = held.groupby(['speaker', 'emotion'])[FEATURES].agg(['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    stats.columns = [f'{feature}_{stat}'.replace('<lambda_0>', 'q25').replace('<lambda_1>', 'q75') for feature, stat in stats.columns]
    stats.to_csv(ROOT / 'Results/acoustic_test_summary.csv')
    shifts = []
    rng = np.random.default_rng(1701)
    for emotion in LABELS:
        sub = held[held.emotion == emotion]
        assert len(sub) == 60 and sub.word.nunique() == 30
        indices = rng.integers(0, 30, size=(2000, 30))
        for feature in ['f0_median_hz', 'rms_dbfs', 'centroid_hz', 'f0_alternative_hz']:
            difference = paired_differences(sub, feature)
            boot = np.median(difference[indices], axis=1)
            shifts.append({'emotion': emotion, 'feature': feature, 'word_pairs': 30, 'median_yaf_minus_oaf': float(np.median(difference)), 'bootstrap_low': float(np.quantile(boot, .025)), 'bootstrap_high': float(np.quantile(boot, .975))})
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(ROOT / 'Results/acoustic_speaker_shifts.csv', index=False)

    cases = json.loads((ROOT / 'Results/failure_cases.json').read_text())
    evidence = []
    indexed = frame.set_index('id')
    for case in cases:
        sample = indexed.loc[case['id']]
        train = frame[frame['split_' + case['protocol']] == 'train']
        if case['protocol'] == 'main':
            train = train[train.speaker == sample.speaker]
        assert sample.word not in set(train.word)
        scales, medians, distances = reference_distances(train, sample)
        true, pred = case['true_label'], case['fusion_prediction']
        ref = {}
        for emotion in [true, pred]:
            subset = train[train.emotion == emotion]
            ref[emotion] = {'n': len(subset), 'median': subset[FEATURES].median().to_dict(),
                            'q25': subset[FEATURES].quantile(.25).to_dict(), 'q75': subset[FEATURES].quantile(.75).to_dict(),
                            'descriptor_distance': float(distances[emotion])}
        evidence.append({'id': case['id'], 'protocol': case['protocol'], 'true_label': true, 'predicted_label': pred,
                         'observed': sample[FEATURES].to_dict(), 'reference_speakers': sorted(train.speaker.unique()),
                         'reference_ids': train.id.tolist(), 'training_iqr_scales': scales.to_dict(), 'references': ref,
                         'closer_to_predicted_than_true': bool(distances[pred] < distances[true]),
                         'nearest_descriptor_class': str(distances.idxmin())})
    write_json('failure_acoustic_evidence.json', evidence)
    prediction_paths = error_accounting(manifest, cfg)

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout='constrained')
    for ax, feature, title in zip(axes, ['f0_median_hz', 'rms_dbfs', 'centroid_hz'], ['Median F0 shift (semitones)', 'RMS level shift (dB)', 'Centroid shift (Hz)']):
        sub = shifts[shifts.feature == feature].set_index('emotion').loc[LABELS]
        point, low, high = (sub[c].to_numpy() for c in ['median_yaf_minus_oaf', 'bootstrap_low', 'bootstrap_high'])
        ax.errorbar(point, np.arange(7), xerr=[point - low, high - point], fmt='o', capsize=3, color='#236d92')
        ax.axvline(0, color='#929fa9', ls='--', lw=1)
        ax.set_yticks(np.arange(7), [e.replace('_', ' ') for e in LABELS] if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel('YAF minus OAF'); ax.grid(axis='x', alpha=.2)
    for suffix in ['png', 'pdf']:
        fig.savefig(ROOT / f'Results/plots/acoustic_speaker_shifts.{suffix}', dpi=180)
    plt.close(fig)
    sensitivity = np.abs(12 * np.log2(frame.f0_alternative_hz / frame.f0_median_hz))
    inputs = [ROOT / 'configs/experiment.json', ROOT / 'data/manifests/manifest.csv', ROOT / 'Results/preprocessing.json', ROOT / 'Results/failure_cases.json', ROOT / 'Results/fusion_comparison.csv', *prediction_paths]
    outputs = ['acoustic_features.csv', 'acoustic_test_summary.csv', 'acoustic_speaker_shifts.csv', 'failure_acoustic_evidence.json', 'error_recurrence.csv', 'error_recurrence_summary.csv', 'fusion_change_events.csv', 'fusion_changes_by_emotion.csv', 'plots/acoustic_speaker_shifts.png', 'plots/acoustic_speaker_shifts.pdf']
    write_json('acoustic_analysis.json', {
        'purpose': 'Post-hoc descriptive analysis after fixed test evaluation; no tuning or retraining.',
        'recordings': len(frame), 'test_recordings': len(held), 'audio_hashes_verified': len(frame),
        'parselmouth_version': parselmouth.__version__, 'praat_version': parselmouth.PRAAT_VERSION,
        'method': {'audio': '16 kHz resample_poly; exact saved classifier trim boundaries; no amplitude normalization.',
                   'pitch': 'Praat raw autocorrelation to_pitch_ac, 10 ms step, 75-600 Hz; other parameters explicit in script. Median and 90th/10th percentile ratio over voiced frames.',
                   'pitch_range_sensitivity': 'Repeat at 60-700 Hz; same raw autocorrelation algorithm. This does not validate pitch against human annotation.',
                   'rms': '20 log10(full-clip RMS / digital full scale); recorded level, not calibrated loudness or vocal effort.',
                   'centroid': 'Mean over nonzero magnitude-STFT frame centroids; 512 FFT, 400-sample periodic Hann, 160 hop, reflect-centered; includes unvoiced regions.',
                   'paired_speaker_shifts': 'Within each emotion, match the same 30 test words; median YAF minus OAF, F0 in semitones. Percentile interval from 2000 bootstrap resamples of word pairs, seed 1701. Conditional on these two actors, not a speaker-population interval; no multiple-testing claim.',
                   'case_distance': 'RMS distance of log-F0 (semitones re 100 Hz), F0 spread, RMS level and centroid to training-class medians, scaled by pooled training IQR. Main references use the same speaker; transfer references use the sole training speaker. Four features fixed before extraction; descriptive, not a trained classifier.'},
        'sensitivity': {'median_absolute_change_st': float(sensitivity.median()), 'max_absolute_change_st': float(sensitivity.max()), 'over_one_semitone': int((sensitivity > 1).sum()), 'test_over_one_semitone': int((sensitivity[frame.split_main == 'test'] > 1).sum())},
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        'source_sha256': {name: digest(ROOT / name) for name in ['scripts/analyze_acoustics.py', 'requirements-analysis.txt']},
        'output_sha256': {name: digest(ROOT / 'Results' / name) for name in outputs},
        'limitations': ['Whole-utterance descriptors omit timing, voice quality and phonetic detail.', 'Pitch can have octave errors; raw-autocorrelation estimates are not ground truth.', 'Speaker, acting style and recording conditions are confounded; no causal attribution.', 'Case selection was pre-existing and error-focused, not a representative sample.']})
    print('Saved acoustic evidence, paired shifts and reconciled error accounting.', flush=True)


if __name__ == '__main__':
    main()
