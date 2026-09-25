"""Pause de l'entraînement depuis l'interface web (controle.json), prise en compte entre deux itérations."""
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from champ_dhonneur.ia.direct import ecrire_phase
from champ_dhonneur.server import spectateur
from champ_dhonneur.server.app import app


def test_api_pause_et_reprise(tmp_path, monkeypatch):
    monkeypatch.setattr(spectateur, "RUNS", tmp_path)
    d = tmp_path / "essai"
    d.mkdir()
    (d / "config.json").write_text("{}")
    c = TestClient(app)
    assert c.get("/api/entrainements/essai").json()["controle"] == {}
    assert c.post("/api/entrainements/essai/pause", json={"pause": True}).json()["controle"]["pause"] is True
    assert json.loads((d / "controle.json").read_text())["pause"] is True
    assert c.get("/api/entrainements/essai").json()["controle"]["pause"] is True
    c.post("/api/entrainements/essai/pause", json={"pause": False})
    assert json.loads((d / "controle.json").read_text())["pause"] is False
    assert c.post("/api/entrainements/inconnu/pause", json={"pause": True}).status_code == 404


def test_entraineur_attend_la_reprise(tmp_path):
    """L'entraîneur en pause signale la phase « pause » et repart dès que la pause est levée."""
    pytest.importorskip("torch")
    from champ_dhonneur.ia.entrainement import Entraineur
    (tmp_path / "controle.json").write_text(json.dumps({"pause": True}))
    faux = SimpleNamespace(dir=tmp_path, etat={"iteration": 7}, cfg=SimpleNamespace(direct=True),
                           torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))

    def reprendre():
        time.sleep(0.5)
        assert json.loads((tmp_path / "direct" / "phase.json").read_text())["phase"] == "pause"
        (tmp_path / "controle.json").write_text(json.dumps({"pause": False}))

    t = threading.Thread(target=reprendre)
    t.start()
    t0 = time.time()
    Entraineur._attendre_reprise(faux)
    t.join()
    assert 0.4 < time.time() - t0 < 5
    ecrire_phase(tmp_path, "autojeu", 8)
    (tmp_path / "controle.json").unlink()
    Entraineur._attendre_reprise(faux)             # sans fichier : aucune attente
