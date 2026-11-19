"""Build the short Assignment 2a report from inspected alignment results."""

import argparse
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path, help="Directory containing the alignment summary and visualization")
parser.add_argument("output", type=Path, help="Destination PDF")
args = parser.parse_args()
source = args.results
target = args.output
with (source / 'alignment_summary.csv').open() as f:
    summary = list(csv.DictReader(f))
with (source / 'boundary_review.csv').open() as f:
    reviewed = list(csv.DictReader(f))

ink = colors.HexColor('#162d3a')
muted = colors.HexColor('#415864')
accent = colors.HexColor('#1c6582')

styles = {
    'label': ParagraphStyle('label', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=accent),
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=ink),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=ink),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14, textColor=ink),
    'note': ParagraphStyle('note', fontName='Helvetica', fontSize=9, leading=12, textColor=muted),
}


def p(text, style='body'):
    return Paragraph(text, styles[style])


def space(height):
    return Spacer(1, height)


rows = [['Word and final phone', 'Full recording', 'Reviewed speech window']]
for row in reviewed:
    raw = f"{row['baseline_start_s']}-{row['baseline_end_s']} s"
    final = f"{row['reviewed_start_s']}-{row['reviewed_end_s']} s"
    rows.append([f"{row['last_word']}  /{row['phone']}/", raw, final])

comparison = Table(rows, colWidths=[176, 156, 196], hAlign='LEFT')
comparison.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7f0f4')),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('TEXTCOLOR', (0, 0), (-1, -1), ink),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ('TOPPADDING', (0, 0), (-1, -1), 7),
    ('LINEBELOW', (0, 1), (-1, -1), 0.35, colors.HexColor('#cbd9df')),
]))

word_count = sum(int(row['words']) for row in summary)
phone_count = sum(int(row['phones']) for row in summary)
short_count = sum(int(row['phones_at_most_20_ms']) for row in summary)

story = [
    p('IIIT HYDERABAD SPEECH PROCESSING LAB  /  ASSIGNMENT 2A', 'label'),
    space(5),
    p('Forced alignment with MFA', 'title'),
    p('Parichay Mago', 'note'),
    space(13),
    p('Method', 'heading'),
    space(3),
    p('Montreal Forced Aligner 3.3.10 aligned six supplied English audio and transcript pairs '
      'with the English MFA acoustic model v3.1.0 and US English MFA dictionary v3.1.0. '
      'I converted the audio to 16 kHz mono PCM WAV and verified that the samples matched the originals. '
      'US English MFA G2P v3.0.0 supplied pronunciations for six missing words, leaving zero unresolved '
      'tokens. Both runs used the dictionary pronunciation of phrase-final <i>bed</i> ending in /d/.'),
    space(12),
    p('Sample alignment', 'heading'),
    space(4),
    Image(str(source / 'sample_alignment.png'), width=528, height=253.44),
    space(2),
    p('<b>Figure 1.</b> Waveform, Praat spectrogram, and MFA word and phone tiers for '
      'ISLE_SESS0131_BLOCKD02_03_sprt1. The empty intervals after <i>bed</i> mark the quiet tail '
      'on the original 4.50 s timeline.', 'note'),
    space(10),
    p('Findings', 'heading'),
    space(3),
    p(f'All {len(summary)} TextGrids opened in Praat via Parselmouth. Their word tiers matched all '
      f'{word_count} prepared transcript tokens, and their phone tiers contained {phone_count:,} labeled '
      'phones. On the full recordings, MFA stretched the final /t/ in <i>bait</i> and /d/ in <i>bed</i> '
      'to the end of the clips. I set speech windows of 0-2.40 s and 0-2.20 s from the waveform and '
      'spectrogram, then realigned both recordings.'),
    space(6),
    comparison,
    space(8),
    p(f'The reviewed /d/ ends at the supplied window limit. Its precise release remains uncertain. '
      f'{short_count} phone intervals in the longer recordings are about 20 ms and merit closer review. '
      'The corpus has no hand-labeled reference boundaries, so these checks establish coverage '
      'and identify timing issues.'),
]

SimpleDocTemplate(str(target), pagesize=letter, leftMargin=42, rightMargin=42,
                  topMargin=34, bottomMargin=32, title='Assignment 2a - Forced alignment with MFA',
                  author='Parichay Mago').build(story)
