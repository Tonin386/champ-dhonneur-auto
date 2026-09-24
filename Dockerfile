# syntax=docker/dockerfile:1
# Les dépendances viennent uniquement du cache local ./wheels (aucun téléchargement pendant la
# construction) : remplir le cache une fois avec « make wheels ». Le contexte nommé « wheels »
# est fourni par docker-compose.yml ; avec docker build : --build-context wheels=./wheels
ARG PIP_LOCAL="--no-index --find-links /wheels -c requirements-docker.txt"

# ---- dépendances (couches mises en cache : un changement du code ne les réinstalle pas) ----
# Les noms reprennent les dépendances de pyproject.toml ; les versions viennent du fichier figé.
FROM python:3.12-slim AS deps
ARG PIP_LOCAL
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements-docker.txt ./
RUN --mount=type=bind,from=wheels,target=/wheels pip install $PIP_LOCAL fastapi uvicorn numpy setuptools

FROM deps AS deps-ia
ARG PIP_LOCAL
RUN --mount=type=bind,from=wheels,target=/wheels pip install $PIP_LOCAL torch tensorboard pytest httpx
# compilateur C pour torch.compile (Triton) : paquets .deb en cache local (./debs, « make debs »)
RUN --mount=type=bind,from=debs,target=/debs dpkg -i /debs/*.deb > /dev/null

# ---- moteur Rust (module champ_rs) : auto-jeu rapide ----------------------
# Dépendances Rust en cache local (rust/vendor) : compilation hors ligne. Même glibc (Debian
# trixie) que python:3.12-slim ; module « abi3 » (Python ≥ 3.12), sans interpréteur à la
# compilation (rust/pyo3-config.txt).
FROM rust:1-slim-trixie AS rust
WORKDIR /rs
COPY rust/ ./
ENV PYO3_CONFIG_FILE=/rs/pyo3-config.txt CARGO_HOME=/tmp/cargo
RUN --mount=type=cache,target=/rs/target cargo build --release --offline \
    && cp target/release/libchamp_rs.so /champ_rs.abi3.so

# ---- base : moteur + interfaces ------------------------------------------
FROM deps AS base
ARG PIP_LOCAL
COPY --from=rust /champ_rs.abi3.so /usr/local/lib/python3.12/site-packages/
COPY pyproject.toml README.md ./
COPY champ_dhonneur ./champ_dhonneur
RUN --mount=type=bind,from=wheels,target=/wheels pip install $PIP_LOCAL .

# ---- test : exécute la suite de tests pendant la construction ------------
FROM base AS test
ARG PIP_LOCAL
COPY tests ./tests
RUN --mount=type=bind,from=wheels,target=/wheels pip install $PIP_LOCAL ".[dev]" && pytest -q

# ---- runtime : interface web légère (bots MCTS / glouton, sans PyTorch) --
FROM base AS runtime
RUN useradd --create-home champ && mkdir -p /data && chown champ /data
USER champ
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/')" || exit 1
CMD ["champ", "serveur", "--port", "8000"]

# ---- ia : PyTorch (CPU ou GPU NVIDIA) pour l'entraînement et le bot IA ---
# Les roues PyPI de PyTorch embarquent CUDA 13 ; seul le pilote NVIDIA de l'hôte (≥ 580)
# est nécessaire (NVIDIA Container Toolkit).
FROM deps-ia AS ia
ARG PIP_LOCAL
COPY --from=rust /champ_rs.abi3.so /usr/local/lib/python3.12/site-packages/
COPY pyproject.toml README.md ./
COPY champ_dhonneur ./champ_dhonneur
COPY configs ./configs
COPY tests ./tests
RUN --mount=type=bind,from=wheels,target=/wheels pip install $PIP_LOCAL ".[ia,dev]"
ENV CHAMP_MODELE=/app/modeles/meilleur.pt
EXPOSE 8000
CMD ["champ", "serveur", "--port", "8000"]
