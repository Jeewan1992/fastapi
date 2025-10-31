# Dockerfile — simple, reliable for Railway
FROM python:3.10-slim

# set working dir
WORKDIR /app

# install build deps for some Python packages (optional but helpful)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libffi-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy dependency file first (to leverage Docker cache)
COPY requirements.txt /app/requirements.txt

# ensure pip, setuptools, wheel are up-to-date and install deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# copy application code
COPY . /app

# expose port (Railway will provide $PORT at runtime)
ENV PORT=8000

# prefer to use env var-driven port in uvicorn command
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
