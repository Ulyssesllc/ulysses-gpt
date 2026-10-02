FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

COPY src/ ./src/
COPY tests/ ./tests/
COPY check_code.sh .
COPY run_project.sh .

RUN chmod +x check_code.sh run_project.sh

RUN ./check_code.sh

CMD ["./check_code.sh"]
