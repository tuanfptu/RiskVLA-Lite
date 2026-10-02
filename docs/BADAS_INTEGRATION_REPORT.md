# BADAS-Open Integration Report

Checked: 2026-10-02

Primary sources:

- [BADAS-Open source, pinned commit](https://github.com/getnexar/BADAS-Open/tree/01368cf5a164a576ae46eae3baf544e4ecdb2946)
- [Official gated model repository](https://huggingface.co/nexar-ai/BADAS-Open)
- [V-JEPA2 base model](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2)

## Access recheck: VERIFIED, body not downloaded

On 2026-10-02 an authenticated HEAD request resolved
`nexar-ai/BADAS-Open` revision `8fda93711e79d72401b0a4efc151b56455885cd2`,
file `weights/badas_open.pth`. The response reported 3,979,436,545 bytes,
which matches the size expected by the adapter. The checkpoint body was not
downloaded. The token was not printed.

| Check | Status |
|---|---|
| Hugging Face authentication | **VERIFIED** |
| `nexar-ai/BADAS-Open` file metadata | **VERIFIED** |
| Checkpoint body download | **NOT ATTEMPTED** |
| GPU inference | **BLOCKED** |

No latency, VRAM, or probability sequence is claimed.

## Runtime status: BLOCKED

`BADASRiskProvider.predict()` fails closed before inference. Its exception
message begins with `BLOCKED` and names every applicable reason:

- `hf_access_unavailable` when no accepted token can resolve the gated checkpoint;
- `gpu_unavailable` when CUDA is requested but no CUDA device is usable;
- `checkpoint_cannot_be_loaded` when the file, official source, or checkpoint
  load is missing or invalid.

No alternate predictor is substituted. The checked-in manifest is
`outputs/blockers/badas_runtime.json`.

The official `weights/badas_open.pth` artifact is auto-gated. File metadata
was resolved with an authenticated HEAD request. This machine has no NVIDIA
runtime, so the body was not downloaded and a real-video smoke test has
**not** been run. No BADAS latency, VRAM, or output values are claimed.

## VERIFIED source contract

The current source:

- decodes an OpenCV-compatible video;
- resamples to approximately 8 FPS;
- uses 16-frame windows with stride one;
- feeds the V-JEPA2 processor, whose effective model input is
  `(batch, time, channels, height, width) = (1, 16, 3, 256, 256)`;
- computes `softmax(logits / 2)[:, 1]`;
- labels class 1 as accident and class 0 as normal;
- returns one float32 score per resampled frame, with a 16-frame `NaN` warm-up.

The output is best described as the model's accident/near-miss class score,
not as a calibrated physical collision probability. The repository's default
`0.8` trigger is an application default, not a validated threshold for the
Nexar action subset.

The checkpoint is a roughly 3.98 GB PyTorch pickle. Loading also requires the
public V-JEPA2 base weights. BADAS code is Apache-2.0; the base model is MIT.

## VERIFIED implementation discrepancies

The current public project is not a drop-in package contract:

1. `BADASModel.predict()` preprocesses a path to a tensor and then passes that
   tensor to a lower-level method that expects a path.
2. Both published CLIs use that inconsistent wrapper.
3. The internally consistent source route is
   `load_badas_model(...).predict(video_path)`.
4. The separately exported `preprocess_video()` returns a channel-first,
   224-pixel tensor that does not match the lower-level prediction route.
5. `albumentations` is imported but missing from the package dependency list.
6. Current source imports V-JEPA2 classes unavailable in the repository's old
   minimum Transformers claim.
7. Checkpoint loading uses `strict=False`; source code only prints missing or
   unexpected keys.
8. Clips with 16 sampled frames or fewer have no finite predictions.

RiskVLA-Lite uses a lazy adapter around the lower-level consistent route and
preserves warm-up `NaN` values. It does not expose these implementation details
to downstream VLA code.

## Wrapper normalization

`BADASRiskProvider` returns:

```text
timestamps: 0, 1/8, 2/8, ...
risk_scores: raw BADAS class-1 scores, including warm-up NaNs
metadata:
  provider/model/source/checkpoint revisions
  score semantics
  target FPS and context window
  device, latency, and peak CUDA memory when measured
```

The wrapper fails if a finite score falls outside `[0, 1]`. Feature extraction
ignores invalid warm-up entries but records their count.

## Required smoke-test evidence

Run:

```bash
HF_TOKEN=... python scripts/run_badas_smoke.py \
  --video /licensed/path/real_video.mp4 \
  --output outputs/badas_smoke.json
```

Pass criteria:

- the pinned official checkpoint is downloaded through normal gate access;
- base and probe load on the recorded device;
- missing/unexpected checkpoint keys are surfaced;
- the video has more than 16 resampled frames;
- output length matches the resampled timeline;
- warm-up values are explicitly invalid;
- all post-warm-up values are finite and in `[0, 1]`;
- measured latency and peak VRAM are recorded.

If current official code cannot load the checkpoint exactly, the smoke test
fails. It must not silently substitute an unrelated predictor or fabricate a
sequence.

## OPEN QUESTIONS

- Whether the current reconstructed attention probe exactly matches the gated
  checkpoint (the model card and current source describe different pooling).
- Exact checkpoint keys and compatibility warnings.
- Calibration and useful validation thresholds on the human Nexar subset.
- Any claim of cross-dataset generalization. Nexar is in-domain for BADAS;
  see `docs/LIMITATIONS.md`.
- Real RTX 3090 latency and memory.

These questions remain open until the official checkpoint is available and the
smoke test artifact is attached.
