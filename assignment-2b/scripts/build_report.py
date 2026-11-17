"""Generate the assignment report exclusively from measured result artifacts."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
INK=colors.HexColor('#183044'); BLUE=colors.HexColor('#236d92'); MUTED=colors.HexColor('#506473')
W=A4[0]-80
STYLES={
    'label':ParagraphStyle('label',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=BLUE,spaceAfter=8),
    'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=INK,spaceAfter=12),
    'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=INK,spaceBefore=10,spaceAfter=6),
    'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.5,leading=13.5,textColor=INK,spaceAfter=8),
    'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8,leading=11,textColor=MUTED,spaceAfter=6),
    'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.3,leading=11,textColor=INK),
}


def p(text,style='body'):
    return Paragraph(text,STYLES[style])


def table(rows,widths):
    t=Table([[p(str(c),'cell') for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0f5')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
                          ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),7),
                          ('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cbd8e1'))]))
    return t


def plot(name,width=W):
    from PIL import Image as PILImage
    path=ROOT/'Results/plots'/f'{name}.png'
    with PILImage.open(path) as im: height=width*im.height/im.width
    return Image(str(path),width=width,height=height)


def page(title):
    return [PageBreak(),p('IIIT HYDERABAD SPEECH PROCESSING LAB / ASSIGNMENT 2B','label'),p(title,'title')]


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d6e0e6'));canvas.line(40,34,A4[0]-40,34)
    canvas.setFont('Helvetica',8);canvas.setFillColor(MUTED)
    canvas.drawString(40,22,'Parichay Mago | TESS multimodal emotion recognition | 17 November 2026')
    canvas.drawRightString(A4[0]-40,22,str(doc.page))


def main():
    audit=json.loads((ROOT/'Results/dataset_audit.json').read_text())
    summary=pd.read_csv(ROOT/'Results/accuracy_table.csv')
    classes=pd.read_csv(ROOT/'Results/per_emotion.csv')
    paired=pd.read_csv(ROOT/'Results/fusion_comparison.csv')
    diagnostics=pd.read_csv(ROOT/'Results/fusion_diagnostics.csv')
    failures=json.loads((ROOT/'Results/failure_cases.json').read_text())
    names={'speech':'Speech','text':'Text','fusion':'Fusion','speech_control':'Speech control'}
    protocols={'main':'Known speakers','oaf_to_yaf':'OAF to YAF','yaf_to_oaf':'YAF to OAF'}
    main_results=summary[summary.protocol=='main'].set_index('pipeline')
    story=[p('IIIT HYDERABAD SPEECH PROCESSING LAB / ASSIGNMENT 2B','label'),
           p('Multimodal emotion recognition','title'),
           p('Parichay Mago | Reproducible comparison of speech, text and fusion','small'),
           p('Objective and measured outcome','heading'),
           p(f'This work implements the three required seven-class emotion classifiers on TESS. On unseen words from the two known speakers, mean test accuracy across three seeds is <b>{100*main_results.loc["speech","accuracy_mean"]:.2f}% for speech</b>, <b>{100*main_results.loc["text","accuracy_mean"]:.2f}% for text</b>, and <b>{100*main_results.loc["fusion","accuracy_mean"]:.2f}% for fusion</b>. Separate speaker-transfer experiments test a harder generalization setting. All model choices were fixed before final test evaluation.'),
           p('Dataset audit','heading'),
           table([['Check','Observed result'],['Archive duplication','5,600 WAV entries; 2,800 byte-identical duplicate entries removed. No duplicate decoded audio across distinct recording IDs.'],
                  ['Coverage','2 speakers x 200 words x 7 emotions = 2,800 unique recordings; 400 per emotion; no missing combinations.'],
                  ['Audio anomalies','2,799 files at 24,414 Hz; OAF_food_fear at 96 kHz. OAF_pad_ps has approximately 0.11% samples at the clipping threshold. Both retained.'],
                  ['Filename repair','OA_bite_neutral.wav is canonicalized to OAF_bite_neutral.wav using its OAF_neutral parent folder. The original archive path and hash are retained.'],
                  ['Transcripts','No transcript files in the Kaggle archive. Reconstructed from the documented carrier phrase and target word. A separate local ASR spot-check matched 16/16 selected examples.']], [105,W-105]),
           Spacer(1,8),plot('dataset_audit'),
           p('Figure 1. Class coverage and original durations after hash-based deduplication. The two actors are identified by the provider codes OAF and YAF. Full audit: Results/dataset_audit.json.','small'),
           p('Dataset: Pichora-Fuller and Dupuis (2020), Toronto emotional speech set, Borealis V1, DOI 10.5683/SP2/E8H2MF. The assignment-specified Kaggle mirror is ejlok1/toronto-emotional-speech-set-tess.','small')]
    story+=page('Architecture decisions')
    story+=[table([['Functional block','Implementation and reason'],
        ['Speech preprocessing','Anti-aliased resampling to mono 16 kHz. Conservative leading/trailing silence trimming with 20 ms RMS frames, a relative 35 dB threshold and 50 ms margin. Preserve amplitude and internal pauses. Explicit padding masks handle varying lengths.'],
        ['Speech feature extraction','64 log-mel bands, 25 ms Hann window, 10 ms hop, 512-point FFT. Training-frame statistics normalize each band. Two temporal CNN layers (64 channels, kernels 5/3, strides 2/2) learn local spectrotemporal patterns and shorten the sequence.'],
        ['Temporal modeling','Bidirectional GRU, 48 units per direction, followed by masked attention pooling. The 96-dimensional representation summarizes changing acoustic cues. Packed sequences and masking prevent padding from influencing the result.'],
        ['Text preprocessing and features','Reconstructed clean transcripts are lowercased and WordPiece-tokenized. No label suffixes, speaker codes or paths enter the model. Frozen DistilBERT token embeddings provide subword features without a new vocabulary for unseen target words.'],
        ['Contextual modeling','The pretrained DistilBERT transformer supplies contextual states. Mean pooling excludes padding and special tokens and gives a 768-dimensional vector. Freezing the encoder reduces target-task overfitting on 200 repetitive sentences.'],
        ['Fusion','Freeze the selected speech encoder and text encoder, standardize embeddings on the training partition, concatenate 96+768 features, then train an 864-to-64-to-7 MLP. This isolates fusion from extra acoustic fine-tuning.'],
        ['Classifier and control','Speech and text use a 64-unit hidden MLP and seven logits. A speech-only 96-to-537-to-7 control has 55,855 head parameters versus 55,815 for fusion, a difference below 0.1%. It controls approximate classifier capacity, not every architectural difference.']], [108,W-108]),
        p('Why text may not help','heading'),
        p('Each reconstructed transcript is repeated with every emotion, twice per class. A deterministic function of text must make the same prediction for those 14 examples and can be correct on only 2. Consequently, the exact text-only accuracy ceiling on every balanced partition is 2/14 = <b>14.2857%</b>. This is a property of the inputs, not evidence that contextual language modeling is generally ineffective.'),
        p('DistilBERT is externally pretrained; TESS is the only labeled emotion dataset used. Frozen revision: 12040accade4e8a0f71eabdb258fecc2e7e948be. Whisper tiny.en is used only for a fallible transcript spot-check, never as classifier input.','small')]
    story+=page('Experimental protocol')
    story+=[p('The same 140/30/30 target-word partition is used for all models and seeds. All recordings of a word remain in one partition. The split seed is 1701; training seeds are 13, 37 and 73.'),
        table([['Protocol','Train','Validation','Test'],['Known speakers / unseen words','1,960; both','420; both','420; both'],['OAF to YAF / unseen words','980; OAF','210; OAF','210; YAF'],['YAF to OAF / unseen words','980; YAF','210; YAF','210; OAF']], [W-240,80,80,80]),
        p('Each speaker-transfer experiment excludes the remaining 1,400 recordings. Validation always comes from the training speaker; the test speaker is unseen. This is a joint change in speaker and target word, rather than a random clip split.'),
        p('Training and checkpoint selection','heading'),
        p('AdamW uses learning rate 0.001, weight decay 1e-4, cross-entropy loss, batch size 64 and gradient norm clipping at 5. Speech training permits 45 epochs; head training permits 60. Early stopping patience is 8. The best validation macro-F1 selects a checkpoint, with validation loss breaking ties. No test result selects an epoch, seed or hyperparameter.'),
        p('All 36 configured runs are included: 3 protocols x 3 seeds x 4 systems. The speech encoder is trained independently for each protocol and seed. Fusion and the capacity control reuse that exact frozen encoder. Feature statistics are fitted exclusively on the training partition.'),
        plot('training_curves'),
        p('Figure 2. Speech training and validation macro-F1 for seed 13, selected in advance. Final tables include all three seeds.','small'),
        p('Integrity checks','heading'),
        p('Automated checks verify complete class coverage, identical duplicate bytes, unique decoded audio, no word/recording/hash overlap between partitions, speaker separation in transfer protocols, training-only normalization, and invariance to extra sequence padding. Feature caches are tied to manifest and preprocessing hashes; fusion checkpoints are tied to their source speech checkpoint.'),
        p('Experiments ran on the local ARM64 CPU with four PyTorch threads. Timing reflects a shared machine, not a controlled hardware benchmark. The environment and implementation snapshot are recorded in Results/run_manifest.json.','small')]
    story+=page('Measured model comparison')
    rows=[['Test protocol','System','Accuracy (%)','Macro-F1 (%)']]
    for protocol in protocols:
        for model in names:
            r=summary[(summary.protocol==protocol)&(summary.pipeline==model)].iloc[0]
            rows.append([protocols[protocol],names[model],f'{100*r.accuracy_mean:.2f} +/- {100*r.accuracy_std:.2f}',f'{100*r.macro_f1_mean:.2f} +/- {100*r.macro_f1_std:.2f}'])
    story+=[table(rows,[130,105,(W-235)/2,(W-235)/2]),Spacer(1,10),plot('accuracy_comparison'),
        p('Table 1 and Figure 3. Mean +/- sample standard deviation across three training seeds on the same test recordings. The dashed line is 1/7 chance accuracy. Because every test set is class-balanced, accuracy and balanced accuracy coincide; the latter is also exported.','small'),
        p('Interpretation','heading')]
    for protocol in protocols:
        s=summary[summary.protocol==protocol].set_index('pipeline')
        delta=100*(s.loc['fusion','accuracy_mean']-s.loc['speech','accuracy_mean'])
        control_delta=100*(s.loc['fusion','accuracy_mean']-s.loc['speech_control','accuracy_mean'])
        story.append(p(f'<b>{protocols[protocol]}:</b> fusion minus speech is {delta:+.2f} percentage points; fusion minus the capacity control is {control_delta:+.2f} points.'))
    story.append(p('Fusion shows no consistent advantage. Its known-speaker mean equals the capacity control, and the speaker-transfer differences are small relative to seed variation. The dominant finding is the generalization gap between known and unseen speakers.'))
    story+=page('Fusion and representations')
    compare=[['Protocol','Fusion fixes / breaks','Shuffled text accuracy','Mean-text accuracy']]
    for protocol in protocols:
        pair=paired[(paired.protocol==protocol)&(paired.reference=='speech')]
        d=diagnostics[diagnostics.protocol==protocol].groupby('condition').accuracy.mean()
        compare.append([protocols[protocol],f'{pair.fusion_fixes.sum()} / {pair.fusion_breaks.sum()}',f'{100*d["shuffled_text"]:.2f}%',f'{100*d["mean_text"]:.2f}%'])
    story+=[table(compare,[125,130,(W-255)/2,(W-255)/2]),
        p('Fixes and breaks are paired recording-level events summed over three seeds, not counts of independent test examples. Shuffled/mean text are evaluation-time perturbations of the trained fusion model, not separately trained ablations. Sensitivity can reflect spurious conditioning or distribution shift.','small'),
        plot('representations'),
        p('Figure 4. Required representations from the speech temporal block, text contextual block and fusion block. Top: emotion; bottom: speaker (OAF blue, YAF red). Separate PCA models are fitted on training embeddings and applied to the same 420 held-out examples. Axes are not comparable across modalities. Seed 13 was fixed before evaluation.','small'),
        p('Identical text embeddings coincide across emotion labels; apparent colours at overlapping points are a rendering effect. Speech and fusion can separate emotion-related structure while also encoding speaker information. Two-dimensional PCA separation is descriptive and does not prove causal emotional features or substitute for held-out classification metrics.')]
    story+=page('Class-level error analysis')
    story+=[plot('confusion_matrices'),p('Figure 5. Row-normalized confusion percentages, accumulated over all three seeds. Rows are true classes; columns are predictions. The same recordings recur across seeds.','small')]
    class_table=[['Protocol / system','Highest recall','Lowest recall']]
    for protocol in protocols:
        for model in ['speech','fusion']:
            sub=classes[(classes.protocol==protocol)&(classes.pipeline==model)].sort_values('recall')
            low,high=sub.iloc[0],sub.iloc[-1]
            class_table.append([protocols[protocol]+' / '+names[model],f'{high.emotion.replace("_"," ")}: {100*high.recall:.1f}%',f'{low.emotion.replace("_"," ")}: {100*low.recall:.1f}%'])
    story+=[table(class_table,[W-280,140,140]),
        p('Table 2. Mean per-class recall across seeds; one class is shown when several tie. The strongest and weakest classes can change with the training speaker. This supports a speaker-dependent error pattern, but the labels and confusion matrices alone do not identify the phonetic cause. Timbre, pitch range and acting style are hypotheses for further listening and annotation, not measured causal conclusions.','small'),
        p('The full precision, recall and F1 table is supplied in Results/per_emotion.csv. The text model provides a controlled demonstration of label ambiguity: the same words correspond to all seven emotions.','small')]
    story+=page('Failure cases and limitations')
    for i,c in enumerate(failures,1):
        neighbours=', '.join(f'{emotion.replace("_"," ")}: {count}' for emotion,count in c['nearest_training_emotions'].items())
        text=(f'<b>{i}. {escape(c["id"])} ({protocols[c["protocol"]]})</b><br/>'
              f'Text: {escape(c["transcript"])}. True: <b>{c["true_label"]}</b>; speech: {c["speech_prediction"]}; '
              f'fusion: {c["fusion_prediction"]} ({100*c["fusion_confidence"]:.1f}% uncalibrated confidence).<br/>'
              f'Five nearest training speech embeddings: {neighbours}. {escape(c["analysis"])}')
        story.append(p(text))
    story+=[p('Limitations and reproducibility','heading'),
        p('TESS contains acted speech from only two speakers and repetitive lexical content. Known-speaker accuracy cannot establish natural-conversation or population-level speaker generalization. Transcript reconstruction was spot-checked by ASR, not exhaustively verified by humans. High-confidence failures are selected for explanation and are not representative prevalence estimates. Seed variation is not a confidence interval over speakers.'),
        p('The public-source package includes separate train.py/test.py entry points for all three required pipelines, the capacity control, setup commands, pinned dependencies and model revision, audit and split manifests, all run metrics, predictions, plots and this report generator. Raw audio is obtained from the provider; it is not redistributed. Checkpoints can be regenerated from the documented commands.'),
        p('References','heading'),
        p('Pichora-Fuller, M. K. and Dupuis, K. (2020). Toronto emotional speech set (TESS), Borealis V1. <link href="https://doi.org/10.5683/SP2/E8H2MF" color="#236d92">doi:10.5683/SP2/E8H2MF</link>.<br/>'
          'Assignment mirror: <link href="https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess" color="#236d92">Kaggle / ejlok1 / TESS</link>.<br/>'
          'Sanh et al. (2019). DistilBERT. <link href="https://arxiv.org/abs/1910.01108" color="#236d92">arXiv:1910.01108</link>. Frozen model revision and full environment are recorded in the repository.','small')]
    doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=45,
                          title='Assignment 2b - Multimodal emotion recognition on TESS',author='Parichay Mago')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(ROOT/'report.pdf')


if __name__=='__main__':
    main()
