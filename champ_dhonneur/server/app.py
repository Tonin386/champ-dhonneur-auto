"""Serveur web : spectateur de l'entraînement (/) et parties Humain / IA avec analyse (/jouer).

L'interface est une application React compilée dans server/web (voir web/ à la racine du
dépôt, « make front ») ; les deux pages partagent le même bundle."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import jeu, spectateur

STATIC = Path(__file__).parent / "static"
WEB = Path(__file__).parent / "web"          # spectateur compilé (make front)
app = FastAPI(title="Champ d'honneur")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(spectateur.router)
app.include_router(jeu.router)
app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")
if (WEB / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.get("/")
@app.get("/jouer")
def page():
    """Spectateur (/) et page « Jouer » : même application React, qui lit l'adresse."""
    if (WEB / "index.html").exists():
        return FileResponse(WEB / "index.html")
    raise HTTPException(503, "Interface web non compilée : lancez « make front »")
