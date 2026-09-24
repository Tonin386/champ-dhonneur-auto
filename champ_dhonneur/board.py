"""Géométrie du plateau de Champ d'honneur.

Le plateau est composé d'hexagones « à sommet plat » disposés en colonnes
verticales. Chaque colonne reçoit une lettre (a, b, c…) de gauche à droite,
et chaque case est numérotée de bas en haut dans sa colonne (1, 2, 3…),
exactement comme les échecs hexagonaux de Glinski.

    Plateau 2 joueurs : colonnes a–g de longueurs 4-5-6-7-6-5-4 (37 cases)
    Plateau 4 joueurs : colonnes a–k de longueurs 2-3-4-5-6-7-6-5-4-3-2 (47 cases)

Le joueur Blanc est en bas (rangs bas), le joueur Noir en haut.

Représentation interne : chaque case a un index entier. On calcule pour
chaque case une coordonnée « double hauteur » (x, y2) où y2 augmente d'une
demi-case entre colonnes voisines et de 2 dans une même colonne, puis des
coordonnées cubiques (q, r, s) pour les distances et les lignes droites.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FILES = "abcdefghijk"

# Vecteurs de voisinage en coordonnées (dx, dy2)
DIRECTIONS = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]


@dataclass
class BoardSpec:
    name: str
    col_lengths: list[int]
    # locations[team] = cases de départ de l'équipe ; neutral = Lieux neutres
    starts: list[list[str]]
    neutral: list[str]
    cells: list[tuple[int, int]] = field(default_factory=list)  # (x, n) n commence à 1
    names: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    xy2: list[tuple[int, int]] = field(default_factory=list)
    cube: list[tuple[int, int, int]] = field(default_factory=list)
    neighbors: list[list[int]] = field(default_factory=list)
    lines2: list[list[tuple[int, int]]] = field(default_factory=list)  # (milieu, bout)
    rays: list[list[list[int]]] = field(default_factory=list)  # par direction
    locations: list[int] = field(default_factory=list)
    dist: list[list[int]] = field(default_factory=list)
    ring2: list[list[int]] = field(default_factory=list)    # cases à distance exactement 2
    within2: list[list[int]] = field(default_factory=list)  # cases à distance 1 ou 2

    def __post_init__(self) -> None:
        maxlen = max(self.col_lengths)
        center = self.col_lengths.index(maxlen)
        pos_of: dict[tuple[int, int], int] = {}
        for x, length in enumerate(self.col_lengths):
            offset = maxlen - length
            for n in range(1, length + 1):
                idx = len(self.cells)
                self.cells.append((x, n))
                name = f"{FILES[x]}{n}"
                self.names.append(name)
                self.index[name] = idx
                y2 = offset + 2 * (n - 1)
                self.xy2.append((x, y2))
                pos_of[(x, y2)] = idx
                q = x
                r = (y2 - x - (center % 2)) // 2
                self.cube.append((q, r, -q - r))
        n_cells = len(self.cells)
        for i in range(n_cells):
            x, y2 = self.xy2[i]
            nb = []
            rays_i = []
            for dx, dy in DIRECTIONS:
                j = pos_of.get((x + dx, y2 + dy))
                if j is not None:
                    nb.append(j)
                ray = []
                k = 1
                while (x + k * dx, y2 + k * dy) in pos_of:
                    ray.append(pos_of[(x + k * dx, y2 + k * dy)])
                    k += 1
                rays_i.append(ray)
            self.neighbors.append(nb)
            self.rays.append(rays_i)
            self.lines2.append([(ray[0], ray[1]) for ray in rays_i if len(ray) >= 2])
        self.dist = [[self._cube_dist(i, j) for j in range(n_cells)] for i in range(n_cells)]
        self.ring2 = [[j for j in range(n_cells) if self.dist[i][j] == 2] for i in range(n_cells)]
        self.within2 = [[j for j in range(n_cells) if 0 < self.dist[i][j] <= 2] for i in range(n_cells)]
        locs = [c for team in self.starts for c in team] + list(self.neutral)
        self.locations = sorted(self.index[c] for c in locs)

    def _cube_dist(self, i: int, j: int) -> int:
        a, b = self.cube[i], self.cube[j]
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def within(self, i: int, d: int) -> list[int]:
        return [j for j in range(self.n_cells) if 0 < self.dist[i][j] <= d]


def board_2p() -> BoardSpec:
    return BoardSpec(
        name="2J",
        col_lengths=[4, 5, 6, 7, 6, 5, 4],
        starts=[["b1", "e1"], ["c6", "f5"]],
        neutral=["a2", "b4", "c3", "e4", "f2", "g3"],
    )


def board_4p() -> BoardSpec:
    return BoardSpec(
        name="4J",
        col_lengths=[2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2],
        starts=[["d1", "g1", "j1"], ["b3", "e6", "h5"]],
        neutral=["a1", "c2", "d4", "e3", "g4", "h2", "i3", "k2"],
    )


_CACHE: dict[str, BoardSpec] = {}


def get_board(name: str) -> BoardSpec:
    if name not in _CACHE:
        _CACHE[name] = board_2p() if name == "2J" else board_4p()
    return _CACHE[name]
