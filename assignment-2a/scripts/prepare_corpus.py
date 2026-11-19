"""Prepare the audio and transcripts in the Assignment 2 ZIP for MFA."""

import argparse
import hashlib
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from zipfile import ZipFile


PART = "Ass_2/Assignment 2a/"
GROUPS = {
    "broadcast": re.compile(r"F2BJRLP[1-3]"),
    "isle": re.compile(r"ISLE_SESS0131_BLOCKD02_0[1-3]_sprt1"),
}
SPEECH_END = {
    "ISLE_SESS0131_BLOCKD02_01_sprt1": 2.40,
    "ISLE_SESS0131_BLOCKD02_03_sprt1": 2.20,
}


def normalize(text: str) -> str:
    text = " ".join(text.lower().split())
    text = text.replace("s.j.c.'s", "s j c's").replace("wbur's", "w b u r's")
    text = text.replace("de-politicize", "depoliticize")
    text = re.sub(r"(?<=\w)-\s*(?=\w)", " ", text)
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text))


def pcm_hash(content: bytes) -> str:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-map", "0:a:0", "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1",
    ]
    decoded = subprocess.run(command, input=content, stdout=subprocess.PIPE, check=True)
    return hashlib.sha256(decoded.stdout).hexdigest()


def prepare(archive_path: Path, corpus: Path, segment_speech: bool) -> None:
    if corpus.exists():
        raise FileExistsError(f"Choose a fresh corpus path: {corpus}")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=corpus.parent) as temp:
        names = archive.namelist()
        sources = sorted(name for name in names if name.startswith(PART + "wav/") and name.endswith(".wav"))
        transcripts = {
            Path(name).stem: name for name in names
            if name.startswith(PART + "transcripts/") and name.lower().endswith(".txt")
        }
        if len(sources) != 6 or {Path(name).stem for name in sources} != transcripts.keys():
            raise ValueError("Expected six matching audio and transcript pairs in Assignment 2a")

        staged = Path(temp) / "corpus"
        for source in sources:
            stem = Path(source).stem
            group = next(group for group, pattern in GROUPS.items() if pattern.fullmatch(stem))
            destination = staged / group / f"{stem}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(source)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 "-map", "0:a:0", "-c:a", "pcm_s16le", str(destination)],
                input=raw, check=True,
            )
            with wave.open(str(destination)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise ValueError(f"Unexpected audio format for {stem}")
                pcm = wav.readframes(wav.getnframes())
                duration = wav.getnframes() / wav.getframerate()
            if hashlib.sha256(pcm).hexdigest() != pcm_hash(raw):
                raise ValueError(f"PCM changed during WAV conversion for {stem}")

            transcript = normalize(archive.read(transcripts[stem]).decode("utf-8-sig"))
            if not transcript:
                raise ValueError(f"Empty transcript for {stem}")
            if segment_speech and stem in SPEECH_END:
                end = SPEECH_END[stem]
                if end >= duration:
                    raise ValueError(f"Speech window extends past the end of {stem}")
                # One labeled utterance and one empty interval on the original time axis.
                annotation = (
                    'File type = "ooTextFile short"\n"TextGrid"\n\n'
                    f'0\n{duration}\n<exists>\n1\n"IntervalTier"\n"{group}"\n'
                    f'0\n{duration}\n2\n0\n{end}\n"{transcript}"\n{end}\n{duration}\n""\n'
                )
                destination.with_suffix(".TextGrid").write_text(annotation, encoding="utf-8")
            else:
                destination.with_suffix(".lab").write_text(transcript + "\n", encoding="utf-8")
            print(f"{group}/{stem}  {duration:.3f} s  {len(transcript.split())} tokens")

        staged.rename(corpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--segment-speech", action="store_true",
                        help="Use Praat-reviewed speech windows for the two recordings with long quiet tails")
    args = parser.parse_args()
    prepare(args.archive, args.corpus.resolve(), args.segment_speech)
