# Mode hybride : jouer sur un vrai plateau contre l'IA

La partie se joue sur la table, avec le matériel. L'écran tient le rôle de l'IA : on y saisit ce
qui se passe sur la table, l'IA répond, et on reproduit ses coups sur le plateau.

Deux exigences :

1. **Reproductibilité** : tout ce qui arrive sur la table doit pouvoir être saisi, et la version
   numérique doit rester identique à la table.
2. **Honnêteté de l'IA** : l'IA ne doit disposer que de ce qu'un vrai joueur assis à sa place
   saurait.

Ce document inventorie ce que ces deux exigences imposent, puis décrit la solution retenue.

## 1. Ce qui se passe pendant une partie

| Événement | Qui le connaît sur la table | Dans le moteur |
|---|---|---|
| Cartes Unité (draft : 8 cartes tirées ; libre : 4 par armée) | tout le monde | mise en place |
| Premier joueur (marqueur Initiative lancé) ou premier à choisir au draft | tout le monde | mise en place |
| Choix d'une carte au draft | tout le monde | décision `draft` |
| Pioche de début de manche (3 pièces, sac vide → défausse remise dans le sac, en cours de pioche) | le joueur seul ; le **nombre** est public | `_start_round` → `_draw` (hasard) |
| Pioche du Moine soldat (1 pièce, même règle de sac vide) | le joueur seul | `_pending_valid` → `_draw` (hasard) |
| Déployer, renforcer, déplacer, attaquer, contrôler, tactique (Sceau royal compris pour la Garde royale) | tout le monde : la pièce est révélée | décision avec sa pièce |
| Passer, recruter, prendre l'Initiative | tout le monde, **sauf la pièce** défaussée face cachée ; la pièce recrutée est montrée | décision avec sa pièce |
| Suites : Berserk, Soldat, Mercenaire, Fantassin, défense de la Garde royale | tout le monde | décisions en attente (`pending`) |
| Remise de la défausse dans le sac, fin de manche, ordre de jeu (un joueur sans pièce est sauté), victoire | tout le monde | automatique |

Chaque décision du moteur est atomique : un geste sur la table correspond donc exactement à une
saisie. Seules les **pioches** sont tirées au hasard par le moteur, et c'est là que la table et
l'écran divergeraient.

## 2. Reproductibilité

### 2.1 Pioches de l'IA : saisies

Les pièces tirées du sac de l'IA doivent être imposées au moteur. Il y a trois moments de pioche :

- le **début de chaque manche**, y compris la première, qu'on joue en mode libre ou après un draft ;
- le **Moine soldat de l'IA** ;
- la **remise en sac en cours de pioche** : si le sac contient moins de pièces qu'il n'en faut,
  les premières pièces viennent de ce qui reste dans le sac, les suivantes de la défausse
  remélangée (livret p. 6).

Conséquences :

- la saisie se fait **pièce par pièce**, dans l'ordre du tirage. À chaque pièce, l'écran propose
  le contenu exact du sac de l'IA (le moteur le connaît), et signale le moment où il faut y
  remettre la défausse ;
- si une seule sorte de pièce est possible, la saisie est **automatique** (dernière pièce du sac,
  par exemple) ;
- en cas de « pas assez de pièces » (moins de 3 au total), la pioche s'arrête d'elle-même.

Mécanisme : le coup qui déclenche une pioche (fin de manche, attaque ou contrôle du Moine soldat,
fin du draft) est d'abord **essayé sur une copie**. La fonction de pioche de cette copie est
remplacée (`_draw`, comme le fait déjà le `Solveur`) : s'il manque une pièce saisie, elle signale
laquelle est attendue, et parmi quelles pièces. Une fois toutes les pièces saisies, le coup est
appliqué pour de bon, avec `force_draws`. Le moteur lui-même n'est pas modifié.

### 2.2 Pioches du joueur plateau : jamais saisies

L'écran ne connaît pas la main du joueur plateau, et ne doit pas la connaître. Or le moteur a
besoin d'une main concrète. Il garde donc une **main fictive**, tirée au hasard comme d'habitude.
Seules comptent les tailles des zones (main, sac, défausse cachée), qui sont publiques et
identiques à la table.

### 2.3 Coups du joueur plateau : « coups libres »

Sur la table, le joueur plateau a pu jouer n'importe quelle pièce qu'il pourrait détenir. Or tout
ce qui ne se trouve ni sur le plateau, ni dans la défausse visible, ni dans la boîte est dans son
sac, sa main ou sa défausse cachée. Ce **multiensemble est public** (pièces de départ, plus les
recrutements, moins les pertes, moins ce qui est visible).

- **Coups proposés** : l'union des coups de chaque pièce possible. Pour les coups face cachée
  (passer, recruter, prendre l'Initiative), la pièce est « cachée » et n'est pas saisie.
- **Pièce révélée** (coup face visible) : elle est échangée avec une pièce de la main fictive,
  prise dans le sac fictif ou la défausse cachée fictive. L'état reste compatible avec tout ce qui
  est public : l'IA ne peut pas distinguer ces répartitions.
- **Moine soldat du joueur plateau** : la pièce piochée est inconnue. Les coups proposés sont ceux
  de toutes les pièces possibles, et la pièce saisie est échangée avec la pièce fictive piochée.
- **Contrôle de cohérence** : une pièce que le joueur plateau ne peut plus avoir (par exemple ses
  deux Soldats sont sur le plateau et il n'en a pas recruté) n'est pas proposée. Si le coup joué
  sur la table n'est pas dans la liste, la table et l'écran ont divergé.

Les suites (Berserk, Soldat, Mercenaire, Fantassin, défense de la Garde royale) et les choix de
cartes du draft sont publics, et se saisissent comme d'habitude.

### 2.4 Historique, annulation, resynchronisation

- **Historique** : l'historique est une suite d'étapes, chacune formée du coup concret et des
  pioches saisies de l'IA. Rejouer depuis le départ est déterministe, car la graine du moteur ne
  sert plus qu'aux pièces fictives et les échanges sont déterministes. « Annuler la saisie » et
  « Reprendre d'ici » fonctionnent donc comme à l'écran.
- **Pioche en cours** : une pioche en cours se recommence sans toucher au coup qui l'a déclenchée.
  Annuler pendant une pioche l'abandonne ; si cette pioche suivait un coup de l'IA, on revient au
  dernier coup saisi du joueur plateau.
- **Divergence non rattrapable** : l'éditeur de position permet de repartir, en hybride, de la
  position de la table. Seuls le total par unité des pièces cachées du joueur plateau (main + sac
  + défausse cachée) et la taille de chaque zone ont un sens ; leur répartition est indifférente.
- **Relevé NCH** : il ne contient pas les pioches, et ne permettrait donc pas de rejouer la partie.
  Il est refusé pour une partie hybride.

## 3. Ce que l'IA sait, et seulement cela

Un joueur assis à la place de l'IA connaît :

- le plateau, les marqueurs, l'Initiative, les réserves, les pièces éliminées et les défausses
  visibles ;
- le **nombre** de pièces en main, dans le sac et dans la défausse cachée adverses, et les pièces
  recrutées (elles sont montrées) ;
- sa propre main, son propre sac et sa propre défausse cachée ;
- par déduction, le **multiensemble** des pièces cachées de l'adversaire.

Il ignore la répartition de ces pièces cachées, les pioches adverses et l'identité des pièces
jouées face cachée.

Ce que la solution garantit :

1. **Rien de caché n'est saisi.** Les pioches du joueur plateau et ses pièces face cachée ne sont
   jamais demandées : cette information n'existe nulle part dans le système, elle ne peut donc
   pas fuiter.
2. **L'IA ne voit pas la main fictive.**
   - Le réseau (`ia/encodage.py`) ne reçoit de l'adversaire que les totaux par unité et les tailles
     des zones. C'est vérifié par `test_pas_de_fuite_d_information`.
   - La recherche (IS-MCTS) redistribue au hasard les pièces cachées adverses à chaque simulation.
   - Le `Solveur` de victoires forcées, avec l'IA comme observatrice, suppose que l'adversaire peut
     avoir n'importe quelle pièce possible.
3. **Correction apportée** (`Game.determinize`) : la redistribution mélangeait les pièces cachées
   à partir de leur ordre dans les listes du moteur. Sans biais en moyenne, la décision de l'IA
   dépendait quand même, concrètement, de la répartition fictive. Les pièces sont maintenant
   triées avant le mélange : **à graine égale, la décision de l'IA ne dépend que de son
   information**, quelle que soit la répartition réelle. C'est vérifié par un test d'invariance.
   La même garantie vaut pour les parties à l'écran contre l'IA.
4. **Pioches futures simulées** : la recherche tire les pioches futures avec son propre
   générateur, jamais avec celui du moteur, qui en partie à l'écran détermine les vraies pioches.
5. **Analyse** : elle est toujours calculée du point de vue de l'IA. Elle est indisponible quand
   le joueur plateau a le trait, car ses « meilleurs coups » seraient ceux d'une main fictive.
6. **Affichage** : la main du joueur plateau et ses pièces face cachée restent masquées, même en
   fin de partie, puisqu'elles sont fictives.

Deux limites, qui ne sont pas des fuites :

- **Inférence** : l'IA ne déduit rien du comportement adverse (« il passe souvent, il a une
  mauvaise main »). C'est la piste « tête de croyance » de `docs/IA.md`.
- **L'opérateur voit la main de l'IA** : il saisit les pioches de l'IA, puis prend dans sa main les
  pièces qu'elle joue face cachée. S'il est aussi le joueur plateau, l'avantage est pour lui, pas
  pour l'IA. Pour une partie à l'aveugle, il faut un tiers pour tenir le sac de l'IA.

## 4. Déroulé d'une manche

1. **Début de manche** : piocher 3 pièces dans le sac de l'IA et les saisir une à une. Le joueur
   plateau pioche les siennes, qui ne sont pas saisies.
2. **Tour de l'IA** : l'écran annonce son coup, avec la pièce à prendre dans sa main ; on le
   reproduit sur la table. Si son Moine soldat attaque ou contrôle, on pioche une pièce dans son
   sac et on la saisit.
3. **Tour du joueur plateau** : il joue sur la table, puis on saisit son coup : la pièce jouée (ou
   « cachée »), puis la case sur le plateau ou le coup dans la liste, puis ses suites éventuelles.
4. Quand toutes les mains sont vides, on revient à l'étape 1.

## 5. Mise en œuvre

- `champ_dhonneur/hybride.py` : pièces possibles, coups libres, concrétisation d'un coup libre,
  pioches imposées essayées sur copie (`TirageRequis`).
- `champ_dhonneur/engine.py` : `determinize` trie les pièces cachées avant de les mélanger (seul
  changement du moteur).
- `champ_dhonneur/server/jeu.py` : sessions hybrides (`hybride: true`, une IA et un joueur
  plateau) ; routes `/tirage` et `/tirage/annuler` ; historique en étapes (coup + pioches).
- `web/src/jouer/` : option « Sur plateau réel » dans « Nouvelle partie » et dans l'éditeur ;
  panneau de pioche de l'IA ; choix de la pièce jouée par le joueur plateau ; coups de l'IA à
  reproduire, affichés sous le plateau.
- `tests/test_hybride.py` : une partie « physique » complète, simulée avec une vraie partie
  cachée, est reproduite par la session hybride (aucun coup réel refusé, état public identique à
  chaque étape) ; décision de l'IA invariante selon la répartition fictive ; annulation et reprise
  déterministes.
