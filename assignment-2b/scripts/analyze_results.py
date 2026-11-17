"""Create tables, paired comparisons, required representation plots and failure cases."""
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tess_emotion.data import LABELS, config, manifest, indices, load_features
from tess_emotion.training import speech_embeddings, raw_features, load_checkpoint, checkpoint_path, head_dimensions
from tess_emotion.networks import Head

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})
COLORS = ["#246a9a", "#d38329", "#24917f", "#9665aa"]
PIPELINES = ["speech", "text", "fusion", "speech_control"]
NAMES = {"speech": "Speech", "text": "Text", "fusion": "Fusion", "speech_control": "Speech control"}
PROTOCOL_NAMES = {"main": "Unseen words, known speakers", "oaf_to_yaf": "OAF training → YAF test", "yaf_to_oaf": "YAF training → OAF test"}
PLOTS = ROOT / "Results/plots"
PLOTS.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(PLOTS / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(PLOTS / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def audit_plot():
    data = pd.read_csv(ROOT / "data/manifests/manifest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    data.groupby(["emotion", "speaker"]).size().unstack().loc[LABELS].plot.bar(ax=axes[0], color=COLORS[:2], rot=35)
    axes[0].set(ylabel="Unique recordings", xlabel="", title="Complete, balanced emotion grid")
    axes[0].set_xticklabels([e.replace("pleasant_surprise", "surprise") for e in LABELS], ha="right")
    for speaker, c in zip(["OAF", "YAF"], COLORS):
        axes[1].hist(data.loc[data.speaker == speaker, "duration_s"], bins=20, alpha=.65, label=speaker, color=c)
    axes[1].set(xlabel="Original duration (s)", ylabel="Recordings", title="Utterance durations")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "dataset_audit")


def analyze():
    cfg = config()
    results = []
    for protocol in cfg["protocols"]:
        for seed in cfg["seeds"]:
            for pipeline in PIPELINES:
                path = ROOT / f"Results/evaluation/{protocol}/seed-{seed}/{pipeline}.json"
                r = json.loads(path.read_text())
                results.append(r)
    flat = pd.DataFrame([{k: r[k] for k in ["protocol", "pipeline", "seed", "n_test", "accuracy", "macro_f1", "balanced_accuracy", "training_seconds", "trainable_parameters", "best_epoch"]} for r in results])
    flat.to_csv(ROOT / "Results/all_runs.csv", index=False)
    summary = flat.groupby(["protocol", "pipeline"], sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), n_test=("n_test", "first"),
        trainable_parameters=("trainable_parameters", "first")).reset_index()
    summary.to_csv(ROOT / "Results/accuracy_table.csv", index=False)
    class_rows = []
    for protocol in cfg["protocols"]:
        for pipeline in PIPELINES:
            group = [r for r in results if r['protocol'] == protocol and r['pipeline'] == pipeline]
            for emotion in LABELS:
                class_rows.append({"protocol": protocol, "pipeline": pipeline, "emotion": emotion,
                                   **{metric: np.mean([r['per_class'][emotion][metric] for r in group]) for metric in ['precision','recall','f1-score']}})
    pd.DataFrame(class_rows).to_csv(ROOT / "Results/per_emotion.csv", index=False)
    paired = []
    for protocol in cfg["protocols"]:
        for seed in cfg["seeds"]:
            frames = {p: pd.read_csv(ROOT / f"Results/predictions/{protocol}-{seed}-{p}.csv").set_index('id') for p in PIPELINES}
            f = frames['fusion']
            for reference in ['speech', 'speech_control']:
                s = frames[reference].loc[f.index]
                sc, fc = s.true_label == s.predicted_label, f.true_label == f.predicted_label
                paired.append({'protocol': protocol, 'seed': seed, 'reference': reference,
                               'fusion_fixes': int((~sc & fc).sum()), 'fusion_breaks': int((sc & ~fc).sum()),
                               'both_wrong': int((~sc & ~fc).sum()), 'accuracy_delta': float(fc.mean()-sc.mean())})
    pd.DataFrame(paired).to_csv(ROOT / 'Results/fusion_comparison.csv', index=False)
    diagnostics = [{"protocol":r['protocol'],"seed":r['seed'],"condition":condition,**values}
                   for r in results if r['pipeline']=='fusion' for condition,values in r['diagnostics'].items()]
    pd.DataFrame(diagnostics).to_csv(ROOT / 'Results/fusion_diagnostics.csv', index=False)
    fig, axes = plt.subplots(1,3,figsize=(10,3.4),sharey=True)
    for ax, protocol in zip(axes,cfg['protocols']):
        sub=summary[summary.protocol==protocol].set_index('pipeline').loc[PIPELINES]
        ax.bar(np.arange(4),100*sub.accuracy_mean,yerr=100*sub.accuracy_std,color=COLORS,capsize=3)
        ax.axhline(100/7, color='#777777',ls='--',lw=.8)
        ax.set_xticks(np.arange(4),['Speech','Text','Fusion','Control'],rotation=25)
        ax.set(title=PROTOCOL_NAMES[protocol],ylim=(0,105))
    axes[0].set_ylabel('Test accuracy (%) · mean ± SD, 3 seeds')
    fig.tight_layout();save(fig,'accuracy_comparison')
    # Sum all three seed matrices; every protocol still evaluates the same test recordings.
    fig,axes=plt.subplots(2,3,figsize=(9.5,6.2))
    for col,protocol in enumerate(cfg['protocols']):
        for row,pipeline in enumerate(['speech','fusion']):
            cm=sum(np.array(r['confusion_matrix']) for r in results if r['protocol']==protocol and r['pipeline']==pipeline)
            rates=cm/cm.sum(1,keepdims=True)
            ax=axes[row,col];ax.imshow(rates,vmin=0,vmax=1,cmap='Blues')
            for i in range(7):
                for j in range(7):
                    if cm[i,j]: ax.text(j,i,f'{100*rates[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if rates[i,j]>.6 else '#222222')
            labels=['Ang','Dis','Fear','Happy','Neut','Surp','Sad']
            ax.set_xticks(range(7),labels,rotation=45);ax.set_yticks(range(7),labels)
            ax.set_title(NAMES[pipeline]+' · '+protocol,fontsize=10)
            ax.set_xlabel('Predicted');ax.set_ylabel('True')
    fig.tight_layout();save(fig,'confusion_matrices')
    # Training curves selected in advance: first seed, no best-seed selection.
    fig,axes=plt.subplots(1,3,figsize=(10,2.8))
    for ax,protocol in zip(axes,cfg['protocols']):
        h=json.loads((ROOT/f'Results/training/{protocol}/seed-13/speech.json').read_text())['history']
        for part,c in [('train',COLORS[0]),('validation',COLORS[2])]:
            ax.plot([r['epoch'] for r in h],[r[part]['macro_f1'] for r in h],label=part,color=c)
        ax.set(title=protocol,xlabel='Epoch',ylim=(0,1.03));ax.legend(fontsize=8)
    axes[0].set_ylabel('Macro-F1');fig.tight_layout();save(fig,'training_curves')
    representation_plots()
    failure_cases()
    audit_plot()
    print(summary.to_string(index=False))


def representation_plots():
    torch.set_num_threads(4)
    data,rows=load_features(),manifest()
    train_ids,test_ids=indices(rows,'main','train'),indices(rows,'main','test')
    speech=speech_embeddings(data,'main',13)
    ck=load_checkpoint(checkpoint_path('main',13,'fusion'))
    model=Head(*head_dimensions('fusion')).eval();model.load_state_dict(ck['state_dict'])
    raw=torch.cat([speech,data['text']],1)
    with torch.inference_mode(): fused=model((raw-ck['mean'])/ck['std'],return_embedding=True)[1]
    arrays=[speech.numpy(),data['text'].numpy(),fused.numpy()]
    names=['Speech temporal representation','Text contextual representation','Fusion representation']
    fig,axes=plt.subplots(2,3,figsize=(10,5.6))
    rng=np.random.default_rng(1701);order=rng.permutation(len(test_ids))
    labels=np.array([rows[i]['label'] for i in test_ids]);speakers=np.array([rows[i]['speaker']=='YAF' for i in test_ids])
    variances=[]
    for col,(x,name) in enumerate(zip(arrays,names)):
        pca=PCA(2,svd_solver='full').fit(x[train_ids]);coords=pca.transform(x[test_ids]);variances.append(pca.explained_variance_ratio_.tolist())
        for row,(colors,cmap) in enumerate([(labels,'tab10'),(speakers,'coolwarm')]):
            axes[row,col].scatter(coords[order,0],coords[order,1],c=colors[order],cmap=cmap,s=10,alpha=.45,vmin=0,vmax=6 if row==0 else 1)
            axes[row,col].set(xlabel='PC1',ylabel='PC2',title=name if row==0 else 'Same points · colour by speaker')
        if col==1: axes[0,col].text(.02,.03,'Identical texts overlap across all emotions',transform=axes[0,col].transAxes,fontsize=7)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',ls='',color=plt.get_cmap('tab10')(i/6),label=LABELS[i].replace('_',' '),markersize=5) for i in range(7)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.055),ncol=4,fontsize=8)
    fig.tight_layout();save(fig,'representations')
    (ROOT/'Results/representation_method.json').write_text(json.dumps({'method':'Separate PCA fitted on training embeddings; same 420 held-out recordings plotted in each panel. Axes are not shared across modalities. Seed 13 was fixed in advance. Points shuffled only for rendering; no label-driven dimensionality reduction.','explained_variance_ratio':dict(zip(names,variances))},indent=2)+'\n')


def failure_cases():
    selected=[]
    # Include both speaker directions; select high-confidence fusion mistakes,
    # while preferring distinct true/predicted pairs. Main seed fixed in advance.
    for protocol in ['main','oaf_to_yaf','yaf_to_oaf']:
        frames={p:pd.read_csv(ROOT/f'Results/predictions/{protocol}-13-{p}.csv').set_index('id') for p in ['speech','fusion']}
        f,s=frames['fusion'],frames['speech'];wrong=f[f.true_label!=f.predicted_label].sort_values('confidence',ascending=False)
        seen=set()
        for identity,r in wrong.iterrows():
            pair=(r.true_label,r.predicted_label)
            if pair in seen:continue
            seen.add(pair)
            speech_pred=s.loc[identity,'predicted_label']
            if speech_pred==r.true_label:
                interpretation='Speech was correct; adding the text-conditioned head introduced this error. This is evidence against assuming fusion always helps.'
            elif speech_pred==r.predicted_label:
                interpretation='Both systems choose the same wrong emotion, consistent with a shared acoustic representation limitation.'
            else:
                interpretation='Both systems are wrong but disagree, showing sensitivity to the classifier and modality combination.'
            if protocol!='main':interpretation+=' The test speaker was absent from training; speaker shift is a plausible contributor, not a proven phonetic cause.'
            selected.append({'protocol':protocol,'id':identity,'transcript':r.transcript,'true_label':r.true_label,'speech_prediction':speech_pred,
                             'fusion_prediction':r.predicted_label,'fusion_confidence':float(r.confidence),'analysis':interpretation})
            if len(seen)>= (1 if protocol=='main' else 2):break
    # If main has zero failures, fill from the held-out speaker cases.
    if len(selected)<5:
        for protocol in ['oaf_to_yaf','yaf_to_oaf']:
            f=pd.read_csv(ROOT/f'Results/predictions/{protocol}-13-fusion.csv')
            s=pd.read_csv(ROOT/f'Results/predictions/{protocol}-13-speech.csv').set_index('id')
            for _,r in f[f.true_label!=f.predicted_label].sort_values('confidence',ascending=False).iterrows():
                if any(x['id']==r.id and x['protocol']==protocol for x in selected):continue
                selected.append({'protocol':protocol,'id':r.id,'transcript':r.transcript,'true_label':r.true_label,'speech_prediction':s.loc[r.id,'predicted_label'],'fusion_prediction':r.predicted_label,'fusion_confidence':float(r.confidence),'analysis':'High-confidence error under an unseen speaker. The paired predictions and acoustic plots support further inspection; no human perceptual relabeling was performed.'})
                if len(selected)>=5:break
            if len(selected)>=5:break
    # Ground the case discussion in held-out representations versus training data.
    data,all_rows=load_features(),manifest()
    by_id={r['id']:i for i,r in enumerate(all_rows)}
    for protocol in {c['protocol'] for c in selected[:5]}:
        vectors=speech_embeddings(data,protocol,13)
        training_ids=indices(all_rows,protocol,'train')
        mean,std=vectors[training_ids].mean(0),vectors[training_ids].std(0).clamp_min(1e-5)
        normalized=(vectors-mean)/std
        for case in [c for c in selected[:5] if c['protocol']==protocol]:
            distance=((normalized[training_ids]-normalized[by_id[case['id']]])**2).sum(1)
            nearest=[training_ids[j] for j in distance.topk(5,largest=False).indices.tolist()]
            votes=dict(Counter(all_rows[j]['emotion'] for j in nearest))
            case['nearest_training_emotions']=votes
            case['nearest_training_ids']=[all_rows[j]['id'] for j in nearest]
            case['neighbour_method']='Five Euclidean nearest training-only speech embeddings, standardized with training means/SDs; descriptive, not a fitted classifier.'
    (ROOT/'Results/failure_cases.json').write_text(json.dumps(selected[:5],indent=2)+'\n')
    # Original audio, not normalized features, for review of the selected failures.
    import soundfile as sf
    rows={r['id']:r for r in manifest()}
    fig,axes=plt.subplots(len(selected[:5]),1,figsize=(10,6.8))
    for ax,case in zip(np.atleast_1d(axes),selected[:5]):
        audio,rate=sf.read(ROOT/rows[case['id']]['audio_path'])
        ax.plot(np.arange(len(audio))/rate,audio,color=COLORS[0],lw=.5)
        ax.set_title(f"{case['id']} · {case['protocol']} · true: {case['true_label']} / fusion: {case['fusion_prediction']}",fontsize=8,loc='left')
        ax.set_ylabel('Amplitude',fontsize=8)
    axes[-1].set_xlabel('Original time (s)');fig.tight_layout();save(fig,'failure_waveforms')


if __name__=='__main__':
    analyze()
