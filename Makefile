.PHONY: install install-ia test web console arene docker entrainer test-ia suivi banc \
        wheels debs verrou images continu continu-arret continu-journal suivi-continu front front-dev

# ---- dépendances Docker en cache local (./wheels) ------------------------------------------
# « make wheels » télécharge une fois pour toutes les versions figées de requirements-docker.txt ;
# les fichiers déjà présents ne sont pas retéléchargés. Les images se construisent ensuite
# sans accès réseau à PyPI.
PY_IMAGE = python:3.12-slim
wheels:
	mkdir -p wheels
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -v "$(CURDIR)/wheels:/wheels" \
	  -v "$(CURDIR)/requirements-docker.txt:/req.txt:ro" $(PY_IMAGE) sh -c '\
	  pip download -q --no-deps --no-index --find-links /wheels -r /req.txt -d /tmp/verif 2>/dev/null \
	    && echo "wheels : cache complet, rien à télécharger" \
	    || pip download --no-deps --progress-bar off -r /req.txt -d /wheels'
# « make debs » met en cache les paquets Debian de gcc (compilateur requis par torch.compile)
debs:
	mkdir -p debs
	@if ls debs/*.deb >/dev/null 2>&1; then echo "debs : cache présent"; else \
	docker run --rm -v "$(CURDIR)/debs:/debs" $(PY_IMAGE) sh -c 'apt-get update -qq && \
	  mkdir -p /tmp/a/partial && apt-get install -y -qq --download-only -o Dir::Cache::archives=/tmp/a gcc && \
	  cp /tmp/a/*.deb /debs/ && chown -R '$$(id -u):$$(id -g)' /debs'; fi
# « make verrou » recalcule les versions (dernières versions compatibles) puis « make wheels »
verrou:
	docker run --rm -v "$(CURDIR):/src:ro" $(PY_IMAGE) sh -c \
	  'cp -r /src /tmp/p && pip install -q "/tmp/p[ia,dev]" && pip freeze | grep -v "^champ-dhonneur"' \
	  | sed '1i # Versions figées des dépendances des images Docker (Python 3.12).\n# Les wheels correspondantes sont mises en cache dans ./wheels par « make wheels ».' \
	  > requirements-docker.txt
	$(MAKE) wheels
images: wheels debs front
	docker compose build web
	docker compose --profile ia build web-ia

# ---- spectateur web (web/ : Vite + React) compilé dans champ_dhonneur/server/web ----------------
# Recompilé seulement si les sources ont changé ; npm local s'il existe, sinon image Docker Node.
# Le premier « npm ci » télécharge les dépendances (web/node_modules), ensuite plus rien.
NODE_IMAGE = node:22-bookworm-slim
WEB_SRC = $(shell find web/src -type f) web/index.html web/package.json web/package-lock.json web/vite.config.ts
champ_dhonneur/server/web/index.html: $(WEB_SRC)
	@if command -v npm >/dev/null 2>&1; then cd web && { [ -d node_modules ] || npm ci; } && npm run build; \
	else docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -v "$(CURDIR):/src" -w /src/web $(NODE_IMAGE) \
	  sh -c '{ [ -d node_modules ] || npm ci; } && npm run build'; fi
front: champ_dhonneur/server/web/index.html
# développement : rechargement à chaud sur http://localhost:5173 (API relayée vers « champ serveur »)
front-dev: ; cd web && npm run dev

# ---- entraînement continu en arrière-plan + interface web (spectateur) -----------------------
continu: wheels debs front
	mkdir -p runs
	docker compose --profile continu up -d --build
continu-arret:
	docker compose --profile continu stop
continu-journal:
	docker compose logs -f --tail 50 entrainement-continu
suivi-continu:
	docker compose run --rm --no-deps entrainement-continu champ suivi runs/continu

# ---- sans Docker ------------------------------------------------------------------------------
install: ; pip install -e ".[dev]"
install-ia: ; pip install -e ".[ia,dev]"
test: ; pytest -q
web: ; champ serveur --port 8000
console: ; champ jouer --blanc humain --noir mcts --unites premiere
arene: ; champ arene mcts glouton --parties 20
docker: ; docker compose build test && docker compose up web
test-ia: ; champ entrainer --config configs/test.json --dossier runs/test
entrainer: ; champ entrainer --config configs/rtx-a5000-laptop.json
banc: ; champ banc --config configs/rtx-a5000-laptop.json
suivi: ; champ suivi runs/principal
