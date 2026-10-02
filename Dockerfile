FROM python:3.12-slim

WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

COPY tests/ ./tests/
COPY check_code.sh .
COPY run_project.sh .

RUN chmod +x check_code.sh run_project.sh

RUN ./check_code.sh

CMD ["./check_code.sh"]
