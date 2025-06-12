ARG PYTHON_VERSION=3.13.3
ARG UV_VERSION=0.7.12

FROM python:${PYTHON_VERSION}-slim

ARG UV_VERSION

LABEL mantainer="Fran <rebelist.software@icloud.com>"

ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color
ENV HF_HUB_CACHE=/home/revelations/.cache
ENV PATH=/home/revelations/.local/bin:$PATH

RUN groupadd --system rebelist && useradd --system --create-home --gid rebelist revelations

USER revelations

WORKDIR /home/revelations/app

COPY --chown=revelations:rebelist ./ .

RUN pip install --no-cache-dir uv==0.7.12 && uv sync --locked --no-dev

ENTRYPOINT ["tail", "-f", "/dev/null"]
