# Notation Champ d'Honneur (NCH)

Une notation compacte, lisible par des humains, calquée sur la notation algébrique des échecs.
Un relevé NCH suffit à rejouer une partie à l'identique.

## Les cases

Le plateau est formé de colonnes verticales d'hexagones. Chaque colonne porte une lettre, de gauche à
droite, et chaque case un numéro de bas en haut dans sa colonne (système des échecs hexagonaux de Glinski).
Or (Blanc dans les relevés et en console, équipe 0) est en bas, Argent (Noir, équipe 1) en haut.

```
Plateau 2 joueurs : colonnes a à g, de hauteurs 4-5-6-7-6-5-4 (37 cases)

              d7
         c6 ◆    e6
    b5        d6       f5 ◆            ◆ = Lieu de départ
a4       c5        e5       g4         ○ = Lieu neutre
    b4 ○      d5       f4
a3       c4        e4 ○     g3 ○
    b3        d4       f3
a2 ○     c3 ○      e3       g2
    b2        d3       f2 ○
a1       c2        e2       g1
    b1 ◆      d2       f1
         c1        e1 ◆
              d1
```

Lieux à 2 joueurs : départs Blanc **b1, e1** ; départs Noir **c6, f5** ; neutres **a2, b4, c3, e4, f2, g3**.

Le plateau 4 joueurs ajoute deux colonnes de chaque côté (a–k). Départs équipe Blanc **d1, g1, j1** ;
équipe Noir **b3, e6, h5** ; neutres **a1, c2, d4, e3, g4, h2, i3, k2**.

Une case est voisine de 6 autres : même colonne ±1, et les colonnes voisines (décalées d'une demi-case).

## Les unités

| Lettre | Unité | | Lettre | Unité |
|---|---|---|---|---|
| A | Archer | | K | Capitaine |
| B | Berserk | | L | Lancier |
| C | Cavalerie | | M | Mercenaire |
| D | Porte étendard (Drapeau) | | N | Chevalier (comme le N des échecs) |
| E | Éclaireur | | P | Piquier |
| F | Fantassin | | R | Moine soldat (Religieux) |
| G | Garde royale | | S | Soldat |
| H | Cavalerie légère (Hussard) | | X | Arbalétrier |
| * | Sceau royal | | | |

Sur un diagramme texte, MAJUSCULE = Blanc, minuscule = Noir, suivie de la hauteur de pile (`S2`, `a1`).

## Les actions

| Action | Notation | Exemple | Lecture |
|---|---|---|---|
| Déployer | `U@case` | `S@b1` | Soldat déployé en b1 (comme le parachutage au shogi) |
| Renforcer | `U+case` | `S+b1` | Soldat en b1 renforcé |
| Déplacer | `Ucase-case` | `Sb1-b2` | |
| Attaquer | `Ucasexcase` | `Sb2xb3` | |
| Contrôler | `Ucase^` | `Sb2^` | on « plante le drapeau » |
| Prendre l'initiative | `I` | `I{A}` | |
| Recruter | `$U` | `$P{X}` | recrute un Piquier en défaussant un Arbalétrier |
| Passer | `--` | `--{*}` | |

`{…}` indique la pièce défaussée face cachée. Il figure dans le **relevé complet** (pour rejouer
exactement la partie) et disparaît du **relevé public** (ce que l'adversaire a vu).

### Tactiques : `U` + case + `:` + détail

| Unité | Exemple | Lecture |
|---|---|---|
| Cavalerie | `Cb2:b3xb4` | va en b3 puis attaque b4 |
| Cavalerie légère | `Hb2:b4` | se déplace de 2 cases |
| Archer | `Ab2:xc4` | tire sur c4 (à 2 cases) |
| Arbalétrier | `Xb2:xb4` | tire en ligne droite |
| Lancier | `Lb2:b4xb5` | charge jusqu'en b4, frappe b5 |
| Capitaine | `Kb2:Sc3xc4` | ordonne au Soldat en c3 d'attaquer c4 |
| Porte étendard | `Db2:Sc3-c4` | fait avancer le Soldat de c3 en c4 |
| Fantassin | `F:>Fb2^>Fd4-d5` | les deux Fantassins manœuvrent |
| Garde royale | `*:Gb2-b3` | le Sceau royal déplace la Garde |

### Effets enchaînés : `>`

Une capacité qui ouvre une nouvelle décision s'écrit à la suite du coup, séparée par `>` :

- Berserk : `Bc3-c4>Bc4xc5>Bc4^` (chaque `>` coûte une pièce de sa pile)
- Soldat : `Sd3xd4>Sd3-e3` (déplacement après l'attaque)
- Moine soldat : `Rb2xb3>H@e1` (la pièce piochée est jouée aussitôt)
- Mercenaire : `$M{S}>Mc3xc4` (manœuvre gratuite au recrutement)
- Garde royale défendue par la réserve : `Sd3xd4(R)`

Une option refusée s'écrit `0` en saisie et n'apparaît pas dans les relevés.

## Relevé de partie (format .nch)

En-têtes entre crochets à la manière du PGN, puis une ligne par manche. Dans une manche, les joueurs
alternent en commençant par le détenteur de l'initiative ; quand un joueur n'a plus de pièce, l'autre
continue seul. L'ordre est donc toujours implicite, comme aux échecs.

```
[Jeu "Champ d'honneur"]
[Mode "2J"]
[Graine "21"]
[Initiative "0"]
[Unites0 "S P X H"]
[Unites1 "A C L E"]
[Resultat "1-0"]

1. S@e1 E@f5 P@b1 C@c6 $H{*} I{A}
2. Ef5-f4 Pb1-b2 L@f5 X@b1 Cc6-b5 $S{H}
3. A@c6 Se1-f1 Lf5-e5 H@e1 $C{*} Xb1-c2
4. Ef4-g3 He1:f2 Le5-e4 Hf2^ Ac6-c5 Pb2-a2
…
1-0
```

La graine fixe les tirages du sac : avec le relevé complet, `champ relire partie.nch` reconstitue la
partie coup par coup.

### Mise en place avancée (draft)

Les 8 cartes tirées et le joueur qui choisit la première sont donnés en en-tête ; les choix forment
la manche `0.`, une lettre par carte, dans l'ordre A1 B2 A2 B2 A1 (A = `PremierChoix`). Le second à
choisir prend l'Initiative (`Initiative`) et commence la manche 1. `Unites0/1` donnent les armées
finales.

```
[Graine "9"]
[Initiative "1"]
[Unites0 "D G M P"]
[Unites1 "B H K X"]
[Draft "B D G H K M P X"]
[PremierChoix "0"]

0. G X B P D H K M
1. B@c6 D@b1 X@f5 G@e1 $K{*} $D{P}
…
```

## Évaluation d'une position

L'analyse (page `/jouer`, `champ analyser`) donne un score à la manière des moteurs d'échecs,
toujours du point de vue d'**Or** :

| Notation | Lecture |
|---|---|
| `+12,4` | Or mène : position aussi favorable qu'environ 1,2 bastion d'avance |
| `−3,0` | Argent mène (léger avantage) |
| `0,0` | équilibre |
| `#3` | victoire forcée d'Or : il pose son dernier marqueur Contrôle en 3 coups au plus |
| `#-2` | victoire forcée d'Argent en 2 coups |
| `1-0`, `0-1`, `½-½` | partie terminée |

**Échelle : 10 points = un bastion (Lieu contrôlé) de plus que l'adversaire.** Le réseau estime
l'espérance de gain v ∈ [-1, 1] ; sur les parties de l'entraînement, une avance de d bastions
donne en moyenne v ≈ tanh(K·d), avec K ≈ 1,05 (un bastion d'avance ≈ 89 % de victoires). Le score
affiché est l'avance en bastions qui donnerait les mêmes chances, multipliée par 10 :
score = 10 · atanh(v) / K. `champ calibrer` recalcule K sur les relevés les plus récents (la
constante est dans `champ_dhonneur/score.py`).

Appréciation, comme aux échecs : `=` égalité (|score| < 3), `⩲` / `⩱` léger avantage Or / Argent
(< 7), `±` / `∓` net avantage (< 15), `+−` / `−+` avantage décisif.

**Victoire forcée.** Un *coup* est une pièce jouée, avec ses effets enchaînés (Berserk, Moine
soldat…), comme un coup aux échecs : `#1` signifie que la pièce qu'on s'apprête à jouer gagne.
La victoire doit tenir contre toutes les défenses et tous les tirages possibles (pièce piochée par
le Moine soldat) ; elle est cherchée jusqu'à la fin de la manche en cours, car au-delà les pièces
piochées dans le sac sont inconnues et rien n'est garanti. Face à une main cachée, l'adversaire
est supposé pouvoir jouer n'importe quelle pièce qu'il pourrait avoir : l'annonce vaut alors
quelle que soit sa main réelle.

La **ligne** qui accompagne chaque coup proposé est la suite la plus explorée par la recherche,
en notation publique, un coup par pièce jouée : `Ed4-c3  Xb1-b2  Ec3^`.

