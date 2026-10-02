# Environment and Compatibility Matrix

Checked: 2026-10-02

## Current development machine — MEASURED

```text
OS: Linux 6.12.94+ x86_64
Python: 3.12.3
NVIDIA tooling: unavailable (`nvidia-smi` not installed)
Repository runtime setup: no preinstalled project environment
```

This machine is suitable for code, deterministic unit tests, annotation
inspection, and artifact review. It cannot verify CUDA model execution.

## Target remote machine

```text
GPU: NVIDIA RTX 3090, 24 GB
OS: Linux
Python: 3.10–3.12
PyTorch: select a current 2.x wheel matching the server driver/CUDA runtime
```

Do not blindly install a CUDA wheel before checking the remote driver's
supported runtime. Record `nvidia-smi`, `torch.__version__`,
`torch.version.cuda`, device name, and BF16 support in every model smoke test.

## Qwen3-VL — VERIFIED source compatibility

| Item | Requirement |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Access | Public, ungated |
| License | Apache-2.0 |
| Transformers | `>=4.57,<6`; Qwen3-VL entered Transformers in 4.57 |
| Weights | One roughly 4.26 GB BF16 safetensors artifact |
| Helper | Native processor needs no `qwen-vl-utils`; helper use requires a Qwen3-VL-compatible release |
| Preferred first run | BF16 if the installed CUDA/PyTorch stack reports support; otherwise FP16 |
| Attention | SDPA first; Flash Attention only after compatible installation and measurement |

The RTX 3090 supports BF16/FP16 tensor operations. Fit and latency for this
project's exact four/eight-frame prompt are still **UNMEASURED**.

## BADAS — VERIFIED source requirements, runtime BLOCKED

Official metadata declares Python 3.8+, PyTorch, Torchvision, Transformers,
Hugging Face Hub, OpenCV, NumPy, Pillow, tqdm, and PyYAML. Current source also
imports `albumentations` without declaring it. Its V-JEPA2 imports require a
substantially newer Transformers version than the old `>=4.40` lower bound.

The project starts from:

```text
transformers >=4.57,<6
torch >=2.4,<3
torchvision >=0.19,<1
albumentations >=1.4,<3
```

These are compatibility ranges, not a claim that every combination has passed
BADAS. Exact versions move into a smoke-test artifact only after successful
official checkpoint execution.

## Reproducible remote setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a PyTorch CUDA wheel appropriate for the server first.
# Then install the project without allowing a silent torch replacement:
python -m pip install -e ".[inference]"

python scripts/inspect_environment.py --output outputs/environment.json
pytest -q
```

For BADAS, pin/clone official source at:

```text
01368cf5a164a576ae46eae3baf544e4ecdb2946
```

and use the gated checkpoint revision recorded in `configs/default.yaml`.
Never commit the clone's checkpoint or Hugging Face token.

## Performance policy

Measure normal BF16/FP16 before 8-bit or 4-bit. Optimization order is:

1. fewer frames;
2. supported resolution reduction;
3. fewer generated tokens;
4. structured action-only output;
5. risk-triggered VLA invocation;
6. compatible fused attention;
7. quantization;
8. LoRA for adaptation, not inference optimization.

No quantization latency claim is made until measured on the actual RTX 3090.
