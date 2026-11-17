"""Train all configured runs, then evaluate the fixed held-out partitions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tess_emotion.data import ROOT, config
from tess_emotion.training import checkpoint_path, load_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=["train", "test", "all"], default="all")
parser.add_argument("--protocols", nargs="+", default=config()["protocols"])
parser.add_argument("--seeds", nargs="+", type=int, default=config()["seeds"])
args = parser.parse_args()
pipelines = ["speech", "text", "fusion", "speech_control"]
for action in (["train", "test"] if args.stage == "all" else [args.stage]):
    for protocol in args.protocols:
        for seed in args.seeds:
            for pipeline in pipelines:
                if action == "train" and checkpoint_path(protocol, seed, pipeline).exists():
                    existing = load_checkpoint(checkpoint_path(protocol, seed, pipeline))
                    if existing["config"] != config():
                        raise RuntimeError("Existing checkpoint has a different config; move it before retraining.")
                    print(f"Resume: existing checkpoint {protocol}/{seed}/{pipeline}", flush=True)
                    continue
                print(f"Running {action}: {protocol}/{seed}/{pipeline}", flush=True)
                command = [sys.executable, "-m", "tess_emotion.training", "--action", action, "--pipeline", pipeline,
                           "--protocol", protocol, "--seed", str(seed)]
                import os
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
                subprocess.run(command, cwd=ROOT, env=env, check=True)
