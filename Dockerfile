# Adaptive LSB Steganography -- container image.
#
# Two stages. The builder installs dependencies into a virtualenv; the
# final image copies only that venv and the source. Build tools, pip
# caches and compilers stay behind, which keeps the shipped image small.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

# Never write .pyc files, never buffer stdout (so docker logs appear live).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements are copied alone, before the source. Docker caches each
# layer, so editing a .py file does not invalidate this one -- the
# dependency install (the slow part) is reused.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH" \
    # Force UTF-8 so revealing Devanagari/CJK/emoji does not blow up on a
    # terminal that defaults to ASCII. Same class of bug as the Windows
    # cp1252 crash that force_utf8_output() in cli.py fixes.
    PYTHONIOENCODING=utf-8

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src/ ./src/
COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY README.md requirements.txt LICENSE ./

# Run as a non-root user. If anything in the container is compromised, it
# does not get root on the container filesystem.
RUN useradd --create-home --uid 1000 stego && \
    chown -R stego:stego /app
USER stego

# Prove the image works at build time. If the round-trip breaks, the
# build fails here instead of shipping a broken image.
RUN python experiments/make_images.py && \
    python -m stego.cli train "images/*.png" --estimators 20 && \
    python -m stego.cli selftest -i images/cover1_landscape.png && \
    python -m pytest tests/ -q

ENTRYPOINT ["python", "-m", "stego.cli"]
CMD ["--help"]
