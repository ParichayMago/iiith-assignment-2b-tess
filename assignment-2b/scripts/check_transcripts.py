"""ASR spot-check of reconstructed text; predictions are diagnostic, never inputs."""
import csv
import json
from pathlib import Path
import sys
import re
import torch
import whisper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tess_emotion.data import manifest

torch.set_num_threads(4)
model = whisper.load_model("tiny.en", device="cpu", download_root=str(ROOT / ".local/whisper"))
rows = [r for r in manifest() if r["word"] == "back" or r["id"] in {"OAF_food_fear", "OAF_pad_ps"}]
out = []
for row in rows:
    prediction = model.transcribe(str(ROOT / row["audio_path"]), fp16=False, language="en", temperature=0.)["text"].strip()
    tokens = re.findall(r"[a-z]+", prediction.lower())
    result = {"id": row["id"], "expected": row["transcript"], "asr": prediction,
              "carrier_found": "say the word" in " ".join(tokens), "target_found": row["word"] in tokens}
    out.append(result)
    print(json.dumps(result), flush=True)
(ROOT / "Results/transcript_spotcheck.json").write_text(json.dumps({"method": "Local Whisper tiny.en, 14 'back' recordings covering both speakers and seven emotions, plus sample-rate and clipping anomalies. ASR is fallible and not a human transcript audit. Predictions never enter classifiers.", "samples": out}, indent=2) + "\n")
