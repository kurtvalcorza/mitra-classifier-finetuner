# DIMER finetuner — Mitra tabular classification (DEFAULT image: CPU).
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV DIMER_TASK_TYPE=tabular_classification
COPY train.py contract_hardening.py contract_entrypoint.py ./
CMD ["python", "contract_entrypoint.py"]
