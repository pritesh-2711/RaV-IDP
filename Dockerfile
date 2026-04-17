FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies.
# libgl1 + libglib2.0-0 + libsm6 + libxext6 + libxrender1 are required by
# opencv-python-headless. tesseract-ocr-eng is the English language pack used
# by pytesseract in the evaluation pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project files first so pip can resolve the package.
COPY . .

# PyTorch CPU-only build.
# Install separately before the project so pip does not pull the default
# CUDA index build when resolving torch as a transitive dependency.
RUN pip install --upgrade pip && \
    pip install \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# transformers: TableTransformer (TATR) used in stage3a evaluation.
# rapidocr: used in utils.py for OCR in comparators; not declared in pyproject.toml.
# Both are runtime requirements that pyproject.toml currently omits.
RUN pip install \
    "transformers>=4.40" \
    "rapidocr>=1.3"

# Install the project and its declared dependencies.
RUN pip install -e ".[dev]"

# Default: drop into a bash shell.
# Override with a specific command when running via docker compose or docker run.
CMD ["bash"]
