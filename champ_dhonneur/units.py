"""Les 16 unités de Champ d'honneur et leur lettre de notation.

Chaque unité est désignée par une lettre majuscule unique, comme les pièces
aux échecs. La pièce Sceau royal est notée « * ».
"""
from __future__ import annotations

from dataclasses import dataclass

ROYAL = "*"


@dataclass(frozen=True)
class UnitDef:
    letter: str
    name: str
    count: int          # nombre total de pièces de ce type
    tactic: str         # description courte de la tactique ("" si aucune)
    ability: str = ""   # capacité (!) ou restriction (x)
    normal_attack: bool = True   # peut effectuer une attaque classique
    max_units: int = 1           # nombre d'unités de ce type simultanément en jeu


UNITS: dict[str, UnitDef] = {u.letter: u for u in [
    UnitDef("A", "Archer", 4,
            "Attaque une unité à 2 cases, non adjacente (la case intermédiaire peut être occupée).",
            "(x) N'attaque qu'avec sa tactique.", normal_attack=False),
    UnitDef("B", "Berserk", 5, "",
            "(!) Après une manœuvre, peut défausser une pièce de sa pile pour manœuvrer à nouveau "
            "(tant qu'il reste au moins 2 pièces)."),
    UnitDef("C", "Cavalerie", 4, "Se déplace puis attaque."),
    UnitDef("D", "Porte étendard", 5,
            "Déplace d'une case une unité alliée à 2 cases ou moins ; elle doit finir à 2 cases ou moins du Porte étendard."),
    UnitDef("E", "Éclaireur", 5, "",
            "(!) Peut être déployé sur une case libre adjacente à une unité alliée."),
    UnitDef("F", "Fantassin", 5, "Chacun des deux Fantassins effectue une manœuvre.",
            "(!) Deux unités de Fantassins peuvent être déployées.", max_units=2),
    UnitDef("G", "Garde royale", 5,
            "Défausse le Sceau royal pour déplacer la Garde royale de 1 ou 2 cases vers un Lieu que "
            "vous contrôlez.",
            "(!) Quand elle est attaquée, peut retirer une pièce de la réserve au lieu de la pile."),
    UnitDef("H", "Cavalerie légère", 5, "Se déplace de 2 cases."),
    UnitDef("K", "Capitaine", 5,
            "Une unité alliée à 2 cases ou moins attaque (attaque classique), si possible."),
    UnitDef("L", "Lancier", 4,
            "Se déplace de 1 ou 2 cases en ligne droite puis attaque dans la même direction.",
            "(x) N'attaque qu'avec sa tactique.", normal_attack=False),
    UnitDef("M", "Mercenaire", 5, "",
            "(!) Quand vous recrutez un Mercenaire et qu'il est déployé, il effectue une manœuvre gratuite."),
    UnitDef("N", "Chevalier", 4, "",
            "(!) Ne peut être attaqué que par des unités renforcées."),
    UnitDef("P", "Piquier", 4, "",
            "(!) Attaqué par une unité adjacente, retire une pièce de l'unité attaquante."),
    UnitDef("R", "Moine soldat", 4, "",
            "(!) Après une attaque ou un contrôle, piochez une pièce et utilisez-la immédiatement."),
    UnitDef("S", "Soldat", 5, "", "(!) Après une attaque, peut se déplacer d'une case."),
    UnitDef("X", "Arbalétrier", 5,
            "Attaque une unité à 2 cases en ligne droite, case intermédiaire libre."),
]}

ALL_LETTERS = sorted(UNITS)

# Mise en place avancée (livret p.12) : nombre de cartes tirées, et ordre des choix en décalage
# par rapport au premier à choisir (A1 B2 A2 B2 A1). 4 joueurs (p.13), non branché :
# 12 cartes, ordre (0, 1, 2, 3, 3, 2, 1, 0, 1, 2, 3, 0) par siège.
DRAFT_POOL = {"2J": 8}
DRAFT_ORDER = {"2J": (0, 1, 1, 0, 0, 1, 1, 0)}

# Répartition conseillée pour la première partie (livret p.5)
FIRST_GAME = [["S", "P", "X", "H"], ["A", "C", "L", "E"]]

# Mises en place historiques (livret p.14-15) : (Blanc, Noir)
BATTLES = {
    "gaugameles": (["N", "H", "P", "K"], ["C", "F", "M", "G"]),
    "bannockburn": (["A", "C", "L", "F"], ["H", "P", "R", "S"]),
    "crecy": (["A", "D", "N", "G"], ["C", "X", "L", "E"]),
}


def unit_name(letter: str) -> str:
    return "Sceau royal" if letter == ROYAL else UNITS[letter].name
