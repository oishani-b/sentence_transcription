import whisper
import torch
from pathlib import Path
import re

# ─── SETUP ────────────────────────────────────────────────────────────────

BASE_DIR    = Path("/Users/monica/Documents/sauce_input")
AUDIO_DIR   = BASE_DIR / "t14"
OUTPUT_DIR  = BASE_DIR / "t14"
# BATCH_SIZE   = 1 
# TARGET_FILE = ".wav"

wav_paths = sorted(AUDIO_DIR.rglob("*.wav"))
print(f"Found {len(wav_paths)} files.")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large").to(DEVICE)
USE_FP16 = (DEVICE == "cuda")

for audio_path in wav_paths:
    print(f"Transcribing: {audio_path.relative_to(AUDIO_DIR)}")

    result = model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        verbose=False,
        fp16=USE_FP16
    )

    text = result["text"].strip()
    # split into short lines (period/question/exclamation)
    split_lines = [s.strip() for s in re.split(r"[\.!?]+", text) if s.strip()]

    # mirror the input directory structure under OUTPUT_DIR, swap .wav -> .txt
    rel_path = audio_path.relative_to(AUDIO_DIR).with_suffix(".txt")
    out_path = OUTPUT_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in split_lines:
            f.write(line + ".\n")

    print(f"Saved: {out_path.relative_to(BASE_DIR)}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\nDONE.")