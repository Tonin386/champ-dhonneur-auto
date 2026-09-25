import pytest


@pytest.fixture(autouse=True)
def historique_temporaire(monkeypatch, tmp_path):
    """Les parties jouées par les tests s'enregistrent dans un dossier temporaire, pas dans ./parties."""
    from champ_dhonneur.server import jeu
    monkeypatch.setattr(jeu, "PARTIES", tmp_path / "parties")
