# DIMER finetuner — Mitra tabular classification (DEFAULT image: CPU).
#
# The default DIMER deployment provisions no GPU node pool (GPU is opt-in and off by default),
# so CPU is the image DIMER should build and run. This slim CPU image also builds comfortably
# within CodeBuild's 15-minute / 2 vCPU / 3 GB limit, which the CUDA image risks exceeding.
#
# Mitra runs on CPU (zero-shot) as well as GPU; fine-tuning itself requires a GPU, so on CPU the
# run is zero-shot. train.py detects the device at runtime (DIMER_TRAIN_DEVICE, including the
# bare "0" form the backend actually emits) and falls back to CPU with a recorded reason.
#
# For a GPU-enabled environment, build Dockerfile.gpu instead — rename it to `Dockerfile` before
# connecting the repo, since DIMER always builds the repository's root `Dockerfile`.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of torch first, from PyTorch's CPU index, so the AutoGluon install
# below does not pull the much larger default CUDA wheel. If pip later resolves a CUDA torch,
# pin a torch version here that satisfies autogluon.tabular==1.5.0.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# OPTION A (recommended for reproducibility): bake pinned Mitra weights into the image so runs
# never depend on a network fetch and cannot drift between builds.
#   ENV HF_HOME=/opt/hf
#   RUN python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('autogluon/mitra-classifier', \
#       revision='c425e9fa0910a6be1c494321792e7ba2a1367b1a')"
# OPTION B: let AutoGluon download Mitra at runtime (simpler, but needs network egress and is
# unpinned unless you also set the revision). Leave as-is to use B.

# Bake our own task type so the Custom/Other -> object_detection normalization can't win.
ENV DIMER_TASK_TYPE=tabular_classification

COPY train.py ./

CMD ["python", "train.py"]
