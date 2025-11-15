ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim

ARG UV_VERSION=0.9
LABEL mantainer="Fran <rebelist.dev@icloud.com>"

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color
ENV HF_HUB_CACHE=/home/revelations/.cache
ENV PATH=/home/revelations/.local/bin:$PATH

RUN groupadd --system rebelist && useradd --system --create-home --gid rebelist revelations

USER revelations

WORKDIR /home/revelations/app

COPY --chown=revelations:rebelist . .

RUN pip install --no-cache-dir "uv~=${UV_VERSION}" && uv sync --locked --no-dev

# Keep the container running indefinitely without consuming CPU.
# This allows fast execution of commands using `docker exec` without restarting the container each time.
ENTRYPOINT ["sleep", "infinity"]
