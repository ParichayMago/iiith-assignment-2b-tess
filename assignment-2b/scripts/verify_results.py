"""Independently reconcile exported predictions, metrics and the frozen source snapshot."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tess_emotion.data import ROOT,LABELS,config


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
    for name,digest in snapshot['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Training source changed: {name}'
    table=pd.read_csv(ROOT/'Results/accuracy_table.csv').set_index(['protocol','pipeline'])
    for (protocol,pipeline),group in pd.DataFrame(checked).groupby(['protocol','pipeline']):
        np.testing.assert_allclose(group.accuracy.mean(),table.loc[(protocol,pipeline),'accuracy_mean'],atol=1e-12)
    failures=json.loads((ROOT/'Results/failure_cases.json').read_text())
    assert 3<=len(failures)<=5
    for case in failures:
        f=pd.read_csv(ROOT/f'Results/predictions/{case["protocol"]}-13-fusion.csv').set_index('id').loc[case['id']]
        assert f.true_label!=f.predicted_label and f.predicted_label==case['fusion_prediction']
    out={'status':'passed','training_source_snapshot_matches':True,'prediction_files_verified':len(checked),
         'checks':['exact test IDs and labels','finite normalized probabilities','confidence and argmax','independent confusion matrix/accuracy/macro-F1','text-only deterministic 1/7 result','summary reconciliation','genuine failure cases'],'runs':checked}
    (ROOT/'Results/verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'PASS: independently verified {len(checked)} prediction files, metrics, summaries and source hashes.')


if __name__=='__main__':main()
