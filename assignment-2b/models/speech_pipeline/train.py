"""Train the speech pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tess_emotion.training import main
if __name__ == "__main__":
    main('speech', 'train')
