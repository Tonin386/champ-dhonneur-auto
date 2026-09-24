# Règles implémentées

Le moteur suit le livret de règles Gigamic (2019) et le texte des 16 cartes Unité (tactiques,
capacités, restrictions et nombre de pièces), vérifié carte par carte sur la planche des cartes.
Les points que ni le livret ni les cartes ne tranchent sont listés en bas de page, avec l'endroit du
code à modifier.

## Déroulement

- Mise en place : 2 pièces de chaque unité + Sceau royal dans le sac, le reste en réserve.
  6 marqueurs Contrôle par joueur (2 déjà placés) ; à 4 joueurs, 8 par équipe (3 placés).
- Chaque manche : chaque joueur pioche 3 pièces (sac vide → la défausse y retourne).
  On joue une pièce à tour de rôle en commençant par le détenteur de l'initiative.
- 9 actions : déployer, renforcer ; initiative, recruter, passer (face cachée) ;
  déplacer, contrôler, attaquer, tactique (face visible).
- Le Sceau royal ne sert qu'aux actions face cachée ou à la tactique de la Garde royale.
- Les pièces retirées par une attaque retournent dans la boîte.
- Victoire immédiate quand le dernier marqueur Contrôle est posé.
- Initiative : impossible si on la détient ou si on l'avait en début de manche ; une fois par manche ;
  jamais prise à son équipier.
- 4 joueurs : contrôle partagé par l'équipe, unités de l'équipier alliées pour les tactiques,
  pas de recrutement pour l'équipier.

## Unités (nombre de pièces)

| | Unité | Pièces | Tactique / capacité |
|---|---|---|---|
| A | Archer | 4 | Attaque à exactement 2 cases, case intermédiaire libre ou non. N'attaque qu'ainsi. |
| B | Berserk | 5 | Après une manœuvre, peut défausser (face visible) une pièce de sa pile pour manœuvrer à nouveau, tant qu'il lui reste au moins 2 pièces. |
| C | Cavalerie | 4 | Se déplace d'une case puis attaque. |
| D | Porte étendard | 5 | Déplace d'une case une unité alliée à ≤ 2 cases ; elle finit à ≤ 2 cases du Porte étendard. |
| E | Éclaireur | 5 | Peut être déployé sur toute case libre adjacente à une unité alliée. |
| F | Fantassin | 5 | Deux unités simultanées ; tactique : chacun des deux manœuvre. |
| G | Garde royale | 5 | Tactique avec le Sceau royal : se déplace de 1 ou 2 cases (chemin libre, pas forcément en ligne droite) vers un Lieu libre contrôlé par son équipe. Attaquée, peut perdre une pièce de la réserve au lieu de sa pile. |
| H | Cavalerie légère | 5 | Se déplace de 2 cases. |
| K | Capitaine | 5 | Une unité alliée à ≤ 2 cases effectue une attaque classique. |
| L | Lancier | 4 | 1 ou 2 cases en ligne droite vers des cases libres, puis attaque obligatoire dans la même direction. N'attaque qu'ainsi. |
| M | Mercenaire | 5 | Recruter un Mercenaire alors qu'il est déployé lui offre une manœuvre gratuite. |
| N | Chevalier | 4 | Ne peut être attaqué que par une unité renforcée. |
| P | Piquier | 4 | Attaqué par une unité adjacente, retire une pièce de l'attaquant. |
| R | Moine soldat | 4 | Après une attaque ou un contrôle, pioche une pièce et la joue aussitôt. |
| S | Soldat | 5 | Après une attaque, peut se déplacer d'une case. |
| X | Arbalétrier | 5 | Attaque à 2 cases en ligne droite, case intermédiaire libre. Attaque classique permise. |

Total : 74 pièces, conforme au matériel.

## Interprétations (validées)

1. **Nombres de pièces** : A4 B5 C4 D5 E5 F5 G5 H5 K5 L4 M5 N4 P4 R4 S5 X5, conformes aux cartes.
   → `units.py`
2. **Moine soldat** : il pioche même s'il vient d'être éliminé par un Piquier. → `engine._trigger`
3. **Soldat / Berserk** : leur capacité ne joue que s'ils survivent à l'attaque. → `engine._pending_valid`
4. **Cavalerie** : la tactique exige une cible (pas de « déplacement seul »). Idem Lancier (FAQ).
5. **Cavalerie légère** : 2 pas successifs par une case libre, arrivée à distance 2.
6. **Capitaine / Porte étendard** ne se ciblent pas eux-mêmes.
7. **Fantassin** : tactique possible seulement avec 2 Fantassins en jeu ; chaque manœuvre est facultative.
8. **Garde royale** : la défense par la réserve est un choix du défenseur (décision intercalée).
9. **Piquier** : pas de riposte contre les tirs à 2 cases (Archer, Arbalétrier).
10. **Berserk** : ses manœuvres supplémentaires sont déplacer, contrôler ou attaquer.
11. **Garde royale (tactique)** : texte de la carte française, « de 1 ou 2 cases vers un Lieu que vous
    contrôlez » ; le chemin peut bifurquer (interprétation courante) mais la case intermédiaire doit être libre.
12. **Mise en place avancée** (`Game(units="draft")`) : 8 cartes tirées au hasard, choix un par un
    (A1 B2 A2 B2 A1, manche 0), puis B prend l'Initiative et commence.
13. **Berserk contre Chevalier** : la pièce de la manœuvre supplémentaire est défaussée avant la
    manœuvre ; le Berserk doit encore être renforcé ensuite pour attaquer un Chevalier.
    → `engine._pending_actions` (`paid=1`), `moteur.rs actions_attente`
14. **4 joueurs** : une capacité déclenchée sur l'unité d'un équipier (Berserk, Soldat, Moine soldat,
    par le Porte étendard ou le Capitaine) est décidée et payée par le propriétaire de l'unité ;
    l'Éclaireur peut être déployé à côté d'une unité de l'équipier. → `engine._trigger`, `_deploy_cells`
15. Garde-fou de simulation : partie nulle après 150 manches (`Game(max_rounds=…)`).
