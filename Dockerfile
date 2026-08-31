FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# CPU-only torch. The default wheel pulls CUDA (~2.5 GB) that this never
# uses -- the corpus was embedded once, offline, on CPU (D-026).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Bake the embedding model into the image (~130 MB). The alternative is
# downloading it on startup, which makes every container start depend on
# huggingface.co being up. Slower builds, predictable starts.
ENV HF_HOME=/app/.hf
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY rag/ ./rag/

# Never root.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
