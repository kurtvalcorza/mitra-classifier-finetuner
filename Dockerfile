# DIMER finetuner — Mitra tabular classification (GPU image, with CPU fallback).
# AutoGluon's Mitra runs on torch/CUDA; use a CUDA runtime base so the 5070 Ti / cluster
# GPU is usable. Match the CUDA minor to the cluster's driver. train.py detects the GPU at
# runtime, so this image also runs on a CPU-only node (zero-shot). For a lean CPU-only image
# instead, use Dockerfile.cpu.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System libs AutoGluon's stack occasionally needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

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
