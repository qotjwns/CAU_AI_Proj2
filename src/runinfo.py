"""
    Where a run happened, recorded with every result so it can be traced back: the device, the commit of the model
    files and the package versions. Used by src/infer_mihbench.py (metrics.json) and src/train.py (args.json).
"""
import datetime
import platform

import torch
import transformers


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_commit(model_dir):
    """The Hugging Face commit the files in model_dir came from (scripts/download_models.py), or None."""
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"  # line 1: commit
    return meta.read_text().splitlines()[0] if meta.exists() else None


def environment(device):
    """Hardware, Python and package versions, and the date."""
    device = torch.device(device)
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "date": datetime.datetime.now().isoformat(timespec="seconds")}
