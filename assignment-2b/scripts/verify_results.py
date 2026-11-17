"""Independently reconcile exported predictions, metrics and the frozen source snapshot."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tess_emotion.data import ROOT,LABELS,config
from tess_emotion.provenance import verify_training_snapshot


def main():
    cfg=config();manifest=pd.read_csv(ROOT/'data/manifests/manifest.csv')
    checked=[]
    for protocol in cfg['protocols']:
        expected=manifest[manifest['split_'+protocol]=='test'].set_index('id')
        for seed in cfg['seeds']:
            for pipeline in ['speech','text','fusion','speech_control']:
                f=pd.read_csv(ROOT/f'Results/predictions/{protocol}-{seed}-{pipeline}.csv').set_index('id')
                assert f.index.is_unique and set(f.index)==set(expected.index)
                assert (f.true_label==expected.loc[f.index,'emotion']).all()
                probs=f[['p_'+e for e in LABELS]].to_numpy()
                assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
                np.testing.assert_allclose(probs.sum(1),1,atol=2e-6)
                assert (np.array(LABELS)[probs.argmax(1)]==f.predicted_label).all()
                np.testing.assert_allclose(probs.max(1),f.confidence,atol=2e-6)
                cm=np.zeros((7,7),dtype=int)
                for actual,predicted in zip(f.true_label,f.predicted_label):cm[LABELS.index(actual),LABELS.index(predicted)]+=1
                accuracy=np.trace(cm)/cm.sum()
                denominator=cm.sum(0)+cm.sum(1)
                macro_f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(7),where=denominator>0).mean()
                r=json.loads((ROOT/f'Results/evaluation/{protocol}/seed-{seed}/{pipeline}.json').read_text())
                np.testing.assert_allclose(accuracy,r['accuracy'],atol=1e-12)
                np.testing.assert_allclose(macro_f1,r['macro_f1'],atol=1e-12)
                np.testing.assert_array_equal(cm,r['confusion_matrix'])
                if pipeline=='text':
                    assert (f.groupby('transcript').predicted_label.nunique()==1).all()
                    np.testing.assert_allclose(accuracy,1/7,atol=1e-12)
                checked.append({'protocol':protocol,'seed':seed,'pipeline':pipeline,'examples':len(f),'accuracy':accuracy})
    snapshot=json.loads((ROOT/'Results/run_manifest.json').read_text())
    amendments=verify_training_snapshot(ROOT,snapshot)
    table=pd.read_csv(ROOT/'Results/accuracy_table.csv').set_index(['protocol','pipeline'])
    for (protocol,pipeline),group in pd.DataFrame(checked).groupby(['protocol','pipeline']):
        np.testing.assert_allclose(group.accuracy.mean(),table.loc[(protocol,pipeline),'accuracy_mean'],atol=1e-12)
    failures=json.loads((ROOT/'Results/failure_cases.json').read_text())
    assert 3<=len(failures)<=5
    for case in failures:
        f=pd.read_csv(ROOT/f'Results/predictions/{case["protocol"]}-13-fusion.csv').set_index('id').loc[case['id']]
        assert f.true_label!=f.predicted_label and f.predicted_label==case['fusion_prediction']
    acoustic=json.loads((ROOT/'Results/acoustic_analysis.json').read_text())
    for section,prefix in [('input_sha256',ROOT),('source_sha256',ROOT),('output_sha256',ROOT/'Results')]:
        for name,digest in acoustic[section].items():
            assert hashlib.sha256((prefix/name).read_bytes()).hexdigest()==digest, f'Acoustic artifact changed: {name}'
    features=pd.read_csv(ROOT/'Results/acoustic_features.csv').set_index('id')
    assert features.index.is_unique and set(features.index)==set(manifest.id)
    assert np.isfinite(features.to_numpy()).all()
    evidence=json.loads((ROOT/'Results/failure_acoustic_evidence.json').read_text())
    assert [c['id'] for c in evidence]==[c['id'] for c in failures]
    indexed=manifest.set_index('id')
    for case in evidence:
        reference=indexed.loc[case['reference_ids']]
        sample=indexed.loc[case['id']]
        assert (reference['split_'+case['protocol']]=='train').all()
        assert sample.word not in set(reference.word)
        if case['protocol']=='main':assert (reference.speaker==sample.speaker).all()
        for name,value in case['observed'].items():
            np.testing.assert_allclose(value,features.loc[case['id'],name],rtol=1e-8)
        for emotion,values in case['references'].items():
            ref=features.loc[reference[reference.emotion==emotion].index]
            assert len(ref)==values['n']==140
            for name,value in values['median'].items():np.testing.assert_allclose(value,ref[name].median(),rtol=1e-8)
    out={'status':'passed','training_source_snapshot_verified':True,'documented_metadata_amendments':amendments,'prediction_files_verified':len(checked),
         'acoustic_recordings_verified':len(features),
         'checks':['exact test IDs and labels','finite normalized probabilities','confidence and argmax','independent confusion matrix/accuracy/macro-F1','text-only deterministic 1/7 result','summary reconciliation','genuine failure cases','acoustic source/input/output hashes','complete finite acoustic measurements','training-only failure references and medians'],'runs':checked}
    (ROOT/'Results/verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'PASS: independently verified {len(checked)} prediction files, metrics, summaries and source hashes.')


if __name__=='__main__':main()
