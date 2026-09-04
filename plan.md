# Plan: Nepali Language Model (Model L) Pipeline Implementation

This document outlines the step-by-step roadmap and todo list to bring the **Nepali (Model L)** language model to feature parity with Hindi (Model H), including ~25M parameter optimization, Hugging Face dataset tokenization/hosting, local checkpointing, and GPU training with auto-recovery on Kaggle.

---

## 1. Context & Key Parameters
- **Vocabulary Size**: 27,998 (from `nepali/tokenizer/artifacts/bpe_ne_32000_vocab.json`)
- **Target Model Size**: ~25M parameters (optimal config: $d_{model}=384, n_{layers}=8, n_{heads}=6, d_{ff}=1536 \rightarrow$ **25.8M parameters** with weight tying)
- **Data Location**:
  - `nepali/data/train.jsonl` (~11.89M lines, ~10 GB)
  - `nepali/data/val.jsonl` (~660K lines, ~569 MB)
  - `nepali/data/test.jsonl` (~660K lines, ~570 MB)
- **Tokenizer Path**: `nepali/tokenizer/artifacts/bpe_ne_32000.pkl`
- **Target Hugging Face Dataset**: `bropal/lma-project-nepali`
- **Target Hugging Face Model Checkpoints**: `bropal/lma-nepali-checkpoints`

---

## 2. Todo List

- [ ] **Step 1: Generalize HF Dataset Tokenization Script**
  - [ ] Update `scripts/push_tokenized_to_hf.py` to accept `--lang ne` (or `--lang hi`).
  - [ ] Set default tokenizer to `nepali/tokenizer/artifacts/bpe_ne_32000.pkl` and data directory to `nepali/data/` when `--lang ne` is specified.
  - [ ] Ensure default repo id resolves to `bropal/lma-project-nepali`.
  - [ ] Verify dry-run with `--max-train-samples 100` on Nepali data.

- [ ] **Step 2: Update Nepali Model Configuration for ~25M Parameters**
  - [ ] Edit `nepali/configs/model_l.yaml`:
    - Set `vocab_size: 27998`
    - Set `d_model: 384`
    - Set `n_layers: 8` (bringing total parameter count to ~25.8M)
    - Set `n_heads: 6` ($d_{head}=64$)
    - Set `d_ff: 1536`
    - Set `context_length: 256`
    - Set `checkpoint.save_dir: "nepali/checkpoints"`
    - Add `checkpoint.hf_repo_id: "bropal/lma-nepali-checkpoints"`

- [ ] **Step 3: Fix Stale Imports and Modularize `nepali/train/`**
  - [ ] Create `nepali/train/checkpoint.py` with Hugging Face download/upload auto-sync (matching Hindi's implementation).
  - [ ] Create `nepali/train/loader.py` for standard text dataset loader without external `phase2/src` dependencies.
  - [ ] Create `nepali/train/model.py` containing helper functions for parameter counting and model creation.
  - [ ] Fix entrypoint in `nepali/train/training.py` to remove legacy `phase2/src` imports and point to `nepali/configs/model_l.yaml`.

- [ ] **Step 4: Create Nepali Kaggle Training Script (`kaggle_train_nepali.py`)**
  - [ ] Configure `kaggle_train_nepali.py` with:
    - Dataset: `bropal/lma-project-nepali`
    - Checkpoints repo: `bropal/lma-nepali-checkpoints`
    - Vocab size: 27,998
    - 25.8M parameter model configuration (`n_layers: 8`)
    - CUDA GPU acceleration and FP16 mixed precision.
    - Checkpoint resumption logic (`step_*.pt` auto-recovery and `RESUME_STEP` support).
    - Auto-uploading of periodic checkpoints and `best_model.pt` to Hugging Face.
    - Post-training inference generator with Nepali sample prompts (e.g., `"नमस्ते"`, `"नेपाल एक सुन्दर દેશ"`).

- [ ] **Step 5: Add CLI Entrypoints to `pyproject.toml`**
  - [ ] Register `lma-train-nepali = "nepali.train.training:main"`
  - [ ] Register `lma-eval-nepali = "nepali.eval.evaluation:main"`

---

## 3. Verification & Validation Steps
1. Run syntax and import test:
   ```bash
   uv run python -c "import nepali.train.training; print('Nepali trainer imported successfully!')"
   ```
2. Test parameter count calculation for Nepali model:
   ```bash
   uv run python -c "
   from nepali.model.model import TransformerLM
   m = TransformerLM(vocab_size=27998, d_model=384, n_layers=8, n_heads=6, d_ff=1536, context_length=256, weight_tying=True)
   total = sum(p.numel() for p in m.parameters() if p.requires_grad)
   print(f'Trainable params: {total:,}')
   "
   ```
3. Test dry run of the Nepali tokenization script:
   ```bash
   uv run python scripts/push_tokenized_to_hf.py --lang ne --max-train-samples 100 --output-dir scratch/nepali_test
   ```
