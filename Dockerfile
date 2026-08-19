# DIMER finetuner — Mitra tabular classification (GPU image, with CPU fallback).
# CUDA 12.8 / torch 2.8 base: ships sm_120 (Blackwell, e.g. RTX 5070 Ti) kernels plus
# sm_70–sm_100, and satisfies AutoGluon 1.5.0's torch>=2.6,<2.10 requirement — so pip does
# NOT silently swap the base image's torch at install time. Do NOT downgrade to
# cuda12.4/torch2.5: that build lacks sm_120 and dies with "no kernel image is available" on
# RTX 50-series / B200 GPUs, and AutoGluon would upgrade torch anyway. train.py detects the
# GPU at runtime, so this image also runs on a CPU-only node (zero-shot); for a lean CPU-only
# image instead, use Dockerfile.cpu.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System libs AutoGluon's stack occasionally needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Lock the actual installed stack: fail the build if pip swapped torch/CUDA out of the tested
# range (AutoGluon manages torch, so this catches a silent dependency drift at build time).
RUN python -c "import torch; v=torch.__version__; c=torch.version.cuda; \
    assert v.startswith('2.8'), 'unexpected torch '+v; \
    assert c and c.startswith('12.8'), 'unexpected CUDA '+str(c); \
    print('locked torch', v, 'CUDA', c)"

# OPTION A (recommended for reproducibility): bake pinned Mitra weights into the image so
# runs never depend on a network fetch and cannot drift between builds.
#   ENV HF_HOME=/opt/hf
#   RUN python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('autogluon/mitra-classifier', \
#       revision='c425e9fa0910a6be1c494321792e7ba2a1367b1a')"
# OPTION B: let AutoGluon download Mitra at runtime (simpler, but needs network egress and
# is unpinned unless you also set the revision). Leave as-is to use B.

# Bake our own task type so the Custom/Other -> object_detection normalization can't win.
ENV DIMER_TASK_TYPE=tabular_classification

COPY train.py ./

CMD ["python", "train.py"]
