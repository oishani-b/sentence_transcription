#!/usr/bin/env python3
"""
Post-process Whisper outputs for SAUCE sentences:

- Convert transcriptions to lowercase and strip punctuation
- Keep only SAUCE files of form:
    sauce_[1–100]_[1–3]_[L1/L2/R1P1/R2P2].wav
- Map each row to a Sentence ID and tag using groundtruth.csv
- Compute per-sentence WER for each model vs gold transcription
- Save a new CSV with added columns:
    sentence_id, tag, reference, tiny_wer, base_wer, ...
"""

import re
import string
from pathlib import Path

import pandas as pd

# ==================== PATHS ====================

BASE_DIR = Path("/Users/monica/Downloads/MTAA")

INPUT_CSV      = BASE_DIR / "combined_whisper_transcription_raw.csv"
GROUNDTRUTH_CSV = BASE_DIR / "groundtruth.csv"
OUTPUT_CSV     = BASE_DIR / "combined_whisper_outputs_sauce_wer.csv"


# Model output columns raw
MODEL_COLS = ["tiny", "base", "small", "medium", "large"]

# ==================== HELPERS ====================

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate (WER)."""
    r = ref.split()
    h = hyp.split()
    N = len(r)

    if N == 0:
        return 0.0 if len(h) == 0 else 1.0

    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost
            )

    return d[-1][-1] / N

# ==================== MAIN ====================

def main():
    # ---- Load data ----
    df = pd.read_csv(INPUT_CSV)
    groundtruth = pd.read_csv(GROUNDTRUTH_CSV)

    # ---- Rename columns for clean merge ----
    df = df.rename(columns={"Sentence ID": "sentence_id"})

    gt = groundtruth.rename(
        columns={
            "sentence_id": "sentence_id",
            "expected_transcription": "reference",
            "tag": "tag",   # high / low prob/nonword
        }
    )

    # ---- Merge ground truth ----
    df = df.merge(gt, on="sentence_id", how="left")

    # ---- Normalize reference + model outputs ----
    text_cols = ["reference"] + [c for c in MODEL_COLS if c in df.columns]
    for col in text_cols:
        df[col] = df[col].apply(normalize_text)

    # ---- Compute WER ----
    for model in MODEL_COLS:
        if model not in df.columns:
            print(f"WARNING: {model} not found; skipping.")
            continue

        wer_col = f"{model}_wer"
        df[wer_col] = df.apply(lambda row: wer(row["reference"], row[model]), axis=1)
        print(f"Mean WER ({model}): {df[wer_col].mean():.3f}")
        
    # ---- Save ----
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved WER output to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()