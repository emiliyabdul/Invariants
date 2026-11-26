"""Download the superconductivity dataset into data/raw/."""
import shutil
import sys
from pathlib import Path

import kagglehub


def download(out_path: str) -> None:
    src_dir = Path(kagglehub.dataset_download("munumbutt/superconductor-dataset"))
    src = src_dir / "train.csv"
    if not src.exists():  # mirror may name files differently; fail with what's there
        raise FileNotFoundError(f"train.csv not in {src_dir}: {[p.name for p in src_dir.iterdir()]}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out_path)


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"
    download(sys.argv[1] if len(sys.argv) > 1 else str(default))
