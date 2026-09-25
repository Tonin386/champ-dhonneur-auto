# Champ d'honneur — moteur, notation et IA

Moteur de règles complet du jeu *Champ d'honneur* (Trevor Benjamin & David Thompson, édition
française de *War Chest*), jouable par des humains en console ou dans le navigateur, avec des bots
classiques et une IA entraînée par apprentissage par renforcement en auto-jeu (voir
[docs/IA.md](docs/IA.md)).

## Démarrage rapide

Avec Docker :

```bash
make continu                          # entraînement sans fin en arrière-plan + spectateur → http://localhost:8090
make continu-journal                  # journal de l'entraînement ; make continu-arret pour l'arrêter

docker compose up web                 # interface web seule → http://localhost:8090
docker compose run --rm console       # partie en console contre le bot MCTS
docker compose run --rm arene         # tournoi MCTS vs glouton, relevés dans ./data/parties
docker compose build test             # construit l'image en exécutant les tests

# IA (image avec PyTorch, GPU NVIDIA)
docker compose run --rm banc          # mesure CPU/GPU et recommande les réglages
docker compose run --rm entrainement  # apprentissage (configs/rtx-a5000-laptop.json) → ./runs/principal
docker compose run --rm evaluation    # meilleur modèle contre le bot MCTS
docker compose --profile ia up web-ia # interface web avec le bot IA (./modeles/meilleur.pt)
```

Les images se construisent **sans téléchargement** à partir de caches locaux, remplis une seule
fois (les cibles `make` ne retéléchargent rien quand le cache est complet) :

| Cache | Contenu | Rempli par |
|---|---|---|
| `wheels/` (2,9 Go) | wheels Python figées dans `requirements-docker.txt` (PyTorch CUDA compris) | `make wheels` (`make verrou` pour recalculer les versions) |
| `debs/` (60 Mo) | paquets Debian de gcc (requis par `torch.compile`) | `make debs` |
| `rust/vendor/` | crates Rust du moteur `champ_rs`, figées dans `rust/Cargo.lock` | `make vendor` |
| `web/node_modules/` | dépendances du spectateur | premier `make front` |

`make continu` et `make images` appellent ces cibles puis construisent les images. Avec
`docker build` directement : `--build-context wheels=./wheels --build-context debs=./debs`.
Le cache Rust est exclu de Git : `make vendor` le prépare après un nouveau clone et le
complète si les dépendances changent. Cette étape utilise Docker, sans installation locale de Rust.

Sans Docker (Python ≥ 3.11) :

```bash
pip install -e ".[dev]"
champ serveur                                         # spectateur (/) et page Jouer (/jouer)
champ jouer --blanc humain --noir mcts --unites premiere
champ jouer --mode 4J --joueurs humain,mcts,glouton,mcts
champ arene mcts:2000 glouton --parties 40 --dossier parties/
champ relire docs/exemple_partie.nch
pytest -q

pip install -e ".[ia]"                                # PyTorch
champ entrainer --config configs/test.json            # vérification rapide (2 min)
champ banc --config configs/rtx-a5000-laptop.json     # calibrage matériel
champ entrainer --config configs/rtx-a5000-laptop.json  # entraînement complet
champ suivi runs/principal                            # tableau de bord
champ jouer --blanc humain --noir ia:400              # jouer contre le réseau
champ analyser partie.nch --coup 30                   # score, victoire forcée, meilleurs coups
champ calibrer runs/continu/parties                   # recalcule l'échelle du score
```

Options d'unités : `premiere` (répartition conseillée du livret), `aleatoire`, ou explicite
`SPXH/ACLE` (Or/Argent, appelés Blanc/Noir en console). Bots : `aleatoire`, `glouton`, `mcts`, `mcts:2000` (itérations),
`mcts:t=3` (secondes par décision), `heur:200` (recherche Gumbel guidée par l'heuristique),
`ia`, `ia:800`, `ia:t=2`, `ia:modele=chemin.pt,sims=400,dispositif=cuda` (réseau entraîné).

## Spectateur grand écran

`http://localhost:8090/` (Docker) ou `http://localhost:8000/` (`champ serveur`) suit en temps réel
l'entraînement le plus récent de `runs/` :

- **direct** : chaque processus d'auto-jeu diffuse une de ses parties en cours (`runs/<nom>/direct/`,
  écrit par `ia/direct.py`, désactivable avec `"direct": false`) ; le navigateur la reçoit coup par
  coup (Server-Sent Events) et l'anime sur le plateau ;
- **mosaïque** de toutes les tables en direct, **rediffusions** des parties enregistrées (auto-jeu et
  évaluations), avec lecture, retour arrière, vitesse et barre de temps ;
- **régie automatique** : suit le direct, enchaîne sur la partie suivante ou sur une rediffusion ;
- tableau de bord : phase en cours et avancement de l'itération, Elo, débit, pertes, précision.

Raccourcis : espace (lecture/pause), ← → (coup par coup), D (direct), M (mosaïque), R (régie),
F (plein écran), 1–9 (table). Survolez le titre d'un graphique pour savoir ce qu'il mesure.

## Jouer, analyser, éditer une position

`/jouer` reprend la disposition du direct (Or à gauche et en bas du plateau, Argent à droite et
en haut) :

- **joueurs** : Humain ou IA entraînée pour chaque camp (Humain contre Humain, Humain contre IA,
  IA contre IA), avec le niveau (simulations par décision) et le modèle (meilleur modèle ou une
  itération précise) ; face à l'IA, sa main et ses pièces jouées face cachée restent secrètes
  (lien « révéler » dans sa colonne) ;
- **mise en place** (« Nouvelle partie », N) : **Draft** par défaut (mise en place avancée,
  8 cartes tirées au hasard ou choisies une à une, premier à choisir au hasard ou imposé),
  **Libre** (les deux armées de 4 unités composées à la main, ou d'après la première partie
  et les batailles historiques ; Initiative au choix) ou **Position** (l'éditeur ci-dessous) ;
- **jeu** : cliquer une pièce de la main puis une case en surbrillance, ou un coup de la liste ;
  « Annuler mon coup », parcours de la partie (← →) et « Reprendre d'ici » pour rejouer une autre
  suite ;
- **analyse** (A) : le réseau évalue la position affichée, à tout moment de la partie, avec une
  barre d'évaluation, ses meilleurs coups et la suite qu'il attend, et une courbe de l'évaluation.
  Face à l'IA, l'analyse n'utilise que l'information du joueur humain ;
- **éditeur de position** (E) : unités de chaque camp, pièces sur le plateau, marqueurs, main,
  sac, défausses, réserve, trait et initiative ; puis « Jouer à partir d'ici » ou « Analyser ».

L'évaluation se lit comme celle d'un moteur d'échecs (détails dans
[docs/NOTATION.md](docs/NOTATION.md#évaluation-dune-position)) : **10 points = un bastion (Lieu
contrôlé) d'avance**, positif quand Or mène, négatif quand Argent mène (`+12,4`, `−3,0`),
symboles `=` `⩲` `±` `+−` ; **`#3`** : Or gagne de force en 3 coups au plus, **`#-2`** : Argent
en 2.

La diffusion en direct fonctionne avec les deux moteurs d'auto-jeu (Python et Rust).

Développement de l'interface : `make front-dev` (rechargement à chaud sur le port 5173, API relayée
vers `champ serveur` sur le port 8000) ; `make front` la recompile dans `champ_dhonneur/server/web`
(fait automatiquement par `make images` et `make continu`).

## Entraînement continu

`make continu` (ou `docker compose --profile continu up -d`) lance deux services redémarrés
automatiquement, même après un redémarrage de la machine :

- `entrainement-continu` : `champ entrainer --config configs/continu.json` → `runs/continu`.
  L'entraînement est sans fin (`"iterations": 0`) mais procède **par itérations** : 2 048 parties
  d'auto-jeu, apprentissage sur la fenêtre d'exemples, puis à chaque itération une évaluation
  (matchs appariés contre le glouton, `heur:64`, la version évaluée précédente et le meilleur
  modèle) qui ajoute un point à la courbe Elo et publie `meilleur.pt`. Le classement
  Bradley-Terry est réajusté sur tous les résultats à chaque évaluation : l'Elo des modèles
  précédents est réévalué et la courbe du spectateur affiche ces valeurs à jour. Il reprend là où
  il s'était arrêté.
- `web-ia` : le spectateur et `/jouer`, dont l'IA suit le meilleur modèle de l'entraînement
  (rechargé dès qu'il change).

L'auto-jeu utilise le **moteur Rust** (`rust/`, module `champ_rs`) : règles, déterminisations,
recherche Gumbel IS-MCTS et encodage y sont portés à l'identique du moteur Python (mêmes listes
de coups légaux dans le même ordre, mêmes pioches pour une même graine, vérifié décision par
décision par `tests/test_rust.py`) ; Python n'évalue que les lots de positions avec le réseau.
Sur la RTX A5000 Laptop, le GPU devient le goulot dès 2 à 3 processus : ≈ 40 000 parties/h
contre ≈ 26 000 avec 11 processus Python. Chaque relevé produit par Rust est rejoué par le
moteur Python avant d'être enregistré (contrôle permanent de cohérence).
`"moteur_autojeu": "python"` revient au moteur Python.

## La notation

Voir [docs/NOTATION.md](docs/NOTATION.md). En bref : `S@b1` déployer, `S+b1` renforcer,
`Sb1-b2` déplacer, `Sb2xb3` attaquer, `Sb2^` contrôler, `Cb2:b3xb4` tactique, `$P` recruter,
`I` initiative, `--` passer, `>` pour les effets enchaînés. Les relevés `.nch` ressemblent au PGN.

## Architecture

```
champ_dhonneur/
  board.py        géométrie hexagonale (2J : 37 cases, 4J : 47), distances, lignes
  units.py        les 16 unités, leur lettre, leurs pièces
  engine.py       état, actions légales, application ; pile de décisions en attente
  notation.py     notation NCH, relevés .nch (export / import / relecture)
  position.py     position quelconque (éditeur) : export, reconstruction validée
  score.py        score de position (10 = un bastion), victoires forcées « #n »
  render.py       plateau en texte
  cli.py          console : jouer, arène, relire, serveur
  arena.py        tournois bot contre bot
  bots/           aléatoire, glouton (heuristique), IS-MCTS, IA (réseau + recherche)
  server/         API FastAPI ; spectateur.py (tableau de bord, direct SSE, rediffusions),
                  jeu.py (parties Humain / IA, analyse, positions), web/ (interface
                  compilée), static/img/ (images)
  ia/
    encodage.py     observation du joueur au trait (46 jetons), actions par caractéristiques
    modele.py       transformeur + têtes valeur (V/N/D) et politique « pointeur »
    evaluateurs.py  évaluation par lots : réseau (CPU/GPU), heuristique, uniforme
    recherche.py    Gumbel IS-MCTS (halving séquentiel, déterminisations, perte virtuelle)
    autojeu.py      parties simultanées, playout cap randomization
    entrainement.py boucle auto-jeu → apprentissage → évaluation, reprenable
    evaluation.py   matchs appariés, classement Elo Bradley-Terry
    analyse.py      analyse de position (score, ligne principale), tableau de bord
    banc.py         banc d'essai matériel (inférence, débit d'auto-jeu, recommandation)
    rs.py           pont vers le moteur Rust (auto-jeu, relevés, direct)
    direct.py       diffusion des parties en cours pour le spectateur
rust/             moteur Rust (PyO3) : règles, encodage, recherche, auto-jeu (module champ_rs)
web/              sources de l'interface (Vite + React + TypeScript) : direct (/) et jouer/
configs/          continu.json, test.json, cpu-demo.json, gpu.json, rtx-a5000-laptop.json
tests/            règles unité par unité, parties aléatoires, notation, IA
docs/             NOTATION.md, REGLES.md, IA.md
```

Choix de conception :

- **Décisions atomiques.** Toute capacité en chaîne (Berserk, Moine soldat, Mercenaire, Soldat,
  Fantassin, défense de la Garde royale) ouvre une décision en attente. Chaque décision a peu
  d'options : c'est plus simple pour l'interface et c'est l'espace d'action idéal pour le RL.
- **Information cachée respectée.** Les bots ne voient que leur camp : `Game.determinize(joueur)`
  redistribue au hasard ce que l'adversaire cache (main, sac, défausse cachée), qui est connu
  exactement en tant que multiensemble.
- **Déterminisme.** Une graine fixe les pioches ; un relevé complet + graine rejoue la partie.
- **Vitesse.** ~90 000 décisions/s en Python pur sur un cœur.

## Force des bots (2 joueurs)

| Match | Résultat |
|---|---|
| glouton vs aléatoire | 20–0 |
| MCTS (800 it.) vs glouton | 5–5 |
| heur:64 vs glouton | 7–5 |

Les bots MCTS et `heur` plafonnent au niveau de l'heuristique qui évalue leurs feuilles. Le bot
`ia` remplace cette heuristique par un réseau appris ; sa force dépend de l'entraînement effectué
(`champ suivi` affiche son Elo, le glouton valant 0).

## Feuille de route

**Phase 1 — fait.** Moteur 2J/4J, notation, console, web, bots de référence, tests, Docker.

**Phase 2 — fait.** Chaîne complète d'apprentissage par renforcement en 2 joueurs : encodage
sans fuite d'information, réseau transformeur, recherche Gumbel IS-MCTS, auto-jeu parallèle,
apprentissage GPU reprenable, évaluation Elo automatique, bot `ia` dans la console et le web
(avec conseil IA), images Docker GPU. Reste à lancer l'entraînement long sur GPU.

**Phase 3 — en cours.** Fait : moteur d'auto-jeu porté en Rust (PyO3), entraînement continu,
spectateur grand écran ; **génération 2** : mise en place avancée (draft) dans le moteur, la
recherche et l'apprentissage (`Game(units="draft")`, 75 % des parties d'auto-jeu), règle de la
Garde royale corrigée (1 ou 2 cases vers un Lieu contrôlé), encodage v2 (cartes disponibles,
pointeur contextuel), têtes auxiliaires, poids moyennés (EMA), évaluations en draft, valeur
dynamique des unités (`runs/<nom>/unites.jsonl`, voir docs/IA.md). La génération 1 est archivée
dans `archives/`. Reste : inférence plus rapide (CUDA graphs, `torch.compile`), évaluations avec le
moteur Rust, solveur exact du draft, vue « Unités » du tableau de bord, 4 joueurs.
