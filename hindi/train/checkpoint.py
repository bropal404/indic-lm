"""Checkpoint detection, resume, and Hugging Face Hub synchronization logic for Trainer."""

import os
import pathlib
from typing import Optional, Dict, Any
import torch


def find_latest_checkpoint(save_dir: str) -> Optional[pathlib.Path]:
    """Find the most recent step checkpoint in the save directory."""
    p = pathlib.Path(save_dir)
    if not p.is_dir():
        return None
    checkpoints = sorted(p.glob("step_*.pt"), key=lambda pp: pp.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def find_best_checkpoint(save_dir: str) -> Optional[pathlib.Path]:
    """Find the best model checkpoint in the save directory."""
    p = pathlib.Path(save_dir)
    best = p / "best_model.pt"
    return best if best.exists() else None


def sync_checkpoints_from_hf(
    repo_id: str,
    save_dir: str,
    token: Optional[str] = None,
) -> Optional[pathlib.Path]:
    """Download checkpoints from Hugging Face Model Hub if present."""
    token = token or os.getenv("HF_TOKEN")
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi(token=token)
        # Check if repo exists
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception:
            print(f"[HF Checkpoint] Repository {repo_id} not found on HF Hub or not accessible. Proceeding with local.")
            return None

        print(f"[HF Checkpoint] Checking for existing checkpoints in HF model repo: {repo_id}...")
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(save_path),
            allow_patterns=["*.pt", "*.yaml", "*.json", "*.txt"],
            token=token,
        )
        print(f"[HF Checkpoint] Successfully synced checkpoints from HF to {save_dir}")
        return find_latest_checkpoint(save_dir)
    except Exception as e:
        print(f"[HF Checkpoint] Note: could not download from HF ({e}). Using local checkpoints.")
        return None


def upload_checkpoint_to_hf(
    file_path: str,
    repo_id: str,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
):
    """Upload a checkpoint file to Hugging Face Model Hub in background/safe block."""
    token = token or os.getenv("HF_TOKEN")
    if not token:
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        # Ensure model repo exists
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        path = pathlib.Path(file_path)
        msg = commit_message or f"Upload checkpoint {path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        print(f"[HF Checkpoint] Synced {path.name} to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[HF Checkpoint] Warning: Failed to sync {file_path} to HF: {e}")


def load_checkpoint_for_resume(
    trainer,
    save_dir: str,
    device: str,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Load latest checkpoint (pulling from HF first if configured) and restore state."""
    if hf_repo_id:
        sync_checkpoints_from_hf(hf_repo_id, save_dir, token=hf_token)

    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path is None:
        print(f"No checkpoints found in {save_dir}. Starting from scratch.")
        return {"step": 0, "epoch": 0, "best_val_loss": float("inf"), "tokens_processed": 0}

    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    trainer.step = checkpoint["step"]
    trainer.epoch = checkpoint["epoch"]
    trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    trainer.tokens_processed = checkpoint.get("tokens_processed", 0)

    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(
        f"Loaded checkpoint from step {checkpoint['step']}, "
        f"epoch {checkpoint['epoch']}, best_val_loss {trainer.best_val_loss:.4f}"
    )

    return {
        "step": trainer.step,
        "epoch": trainer.epoch,
        "best_val_loss": trainer.best_val_loss,
        "tokens_processed": trainer.tokens_processed,
    }
