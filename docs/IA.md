# L'IA de Champ d'honneur

Le bot fort est obtenu par **apprentissage par renforcement en auto-jeu**, dans l'esprit
d'AlphaZero, avec plusieurs techniques plus récentes adaptées à un jeu à information cachée.
Tout le code est dans `champ_dhonneur/ia/` ; le bot jouable est `champ_dhonneur/bots/neural.py`.

```
            ┌───────────── auto-jeu (N processus, CPU ou GPU) ─────────────┐
 réseau ──▶ │ parties simultanées → recherche Gumbel IS-MCTS → exemples   │ ──▶ fichiers .npz
   ▲        └─────────────────────────────────────────────────────────────┘        │
   │                                                                               ▼
   └──── apprentissage (GPU, bf16) sur une fenêtre glissante d'exemples ◀── fenêtre
                         │
                         └──▶ évaluation périodique (matchs appariés, Elo) → meilleur.pt
```

## Démarrer un entraînement

Sur la machine avec le GPU NVIDIA (pilote + NVIDIA Container Toolkit) :

```bash
docker compose build entrainement
docker compose run --rm banc                  # mesure la machine, recommande les réglages
docker compose run --rm entrainement          # configs/rtx-a5000-laptop.json → ./runs/principal
CHAMP_CONFIG=configs/gpu.json docker compose run --rm entrainement   # autre configuration
```

Sans Docker :

```bash
pip install -e ".[ia,dev]"
champ entrainer --config configs/gpu.json
champ entrainer --config configs/gpu.json --set travailleurs=6 --set autojeu.simulations=128
```

**Robustesse.** Les processus d'auto-jeu sont persistants (CUDA n'est initialisé qu'une fois,
le réseau n'est rechargé que s'il a changé) et s'arrêtent d'eux-mêmes si l'entraîneur est tué.
Les exemples sortis de la fenêtre et les modèles non évalués sont supprimés au fil de l'eau
(`purger_donnees`, `purger_modeles`) pour ne pas remplir le disque.

**Reprise.** Relancer la même commande reprend à la dernière itération terminée (modèle,
optimiseur, fenêtre d'exemples, classement Elo). `Ctrl-C` abandonne seulement l'itération en
cours. Pour reprendre, `champ entrainer --dossier runs/principal` suffit : la configuration
enregistrée dans le dossier est réutilisée, les `--set` s'appliquent par-dessus.

**Réglages à adapter à votre machine** (dans `configs/gpu.json` ou via `--set`) :

| Réglage | Conseil |
|---|---|
| `travailleurs` | `-1` (automatique : cœurs physiques − 1). Le moteur Python est le facteur limitant de l'auto-jeu. |
| `dispositif_autojeu` | `cuda` (chaque processus charge le réseau sur le GPU, ≈ 0,5 Go de VRAM chacun) ; `cpu` si la VRAM manque. |
| `autojeu.simultanees` | parties jouées en même temps par processus = taille des lots GPU (32 à 128). |
| `parties_par_iteration` | 500 à 1 000 : une itération dure alors quelques minutes. |
| `lot` | 1 024 pour le réseau de 6,5 M paramètres ; réduire si la mémoire GPU est insuffisante. |

Pour vérifier l'installation en deux minutes : `champ entrainer --config configs/test.json`.
`configs/cpu-demo.json` est une petite configuration qui apprend réellement sur CPU.

## RTX A5000 Laptop : configuration dédiée

`configs/rtx-a5000-laptop.json` est la configuration par défaut de `docker compose run --rm
entrainement`. Ses choix, et comment les ajuster :

**Le CPU fixe le débit, pas le GPU.** Chaque processus d'auto-jeu fait tourner le moteur Python
sur un cœur. Le nombre de processus est donc automatique (`travailleurs: -1` = cœurs physiques
− 1, soit 7 sur un portable à 8 cœurs), et la mémoire du GPU (16 Go) n'est pas une contrainte.
Chaque processus garde son contexte CUDA, environ 0,3 à 0,5 Go.

**Réseau compact (2,1 M paramètres, d = 160, 6 couches).** Sept processus qui interrogent le même
GPU font de l'ordre de 50 000 à 70 000 évaluations par seconde. Le réseau de 7 M paramètres de `gpu.json`
demanderait environ 45 TFLOP/s, hors de portée d'un GPU portable de 90 W, et ralentirait tout
l'auto-jeu. Le réseau de 2,1 M en demande environ 13, ce qui laisse de la marge. Pour un plateau
de 37 cases, c'est largement suffisant au début. Si `champ banc` indique que le GPU a de la
marge, passez à `d = 192` (3 M) en cours de route en démarrant un nouveau dossier.

**Lots.** 96 parties simultanées par processus, et 2 048 parties par itération : chaque processus
en joue environ 290, soit 3 vagues. Si vous avez plus de cœurs, le nombre de parties
simultanées est réduit automatiquement pour garder 3 vagues.

**Précision.** Ampere gère bf16 : les poids d'inférence sont convertis en bf16 et l'apprentissage
se fait en bf16 mixte, avec TF32 pour le reste.

**Durée.** Une itération dure de 2 à 5 minutes selon le CPU. 200 itérations représentent
environ 400 000 parties, soit une à deux nuits. L'évaluation, toutes les 5 itérations, joue
3 × 64 parties réparties sur tous les processus.

### Avant le premier entraînement

```bash
docker compose build entrainement
docker compose run --rm banc          # ~5 minutes : mesure et recommandation
docker compose run --rm entrainement
```

`champ banc` affiche le débit d'inférence du GPU selon la taille des lots (réseau compilé si
`inference_compilee` est vrai ; `--modele runs/continu/modeles/meilleur.pt` joue l'essai d'auto-jeu
avec un réseau entraîné, dont les parties et les lots sont représentatifs). Il affiche aussi le
débit réel de l'auto-jeu pour plusieurs nombres de processus, puis recommande un nombre de
processus et estime la charge du GPU :

- **charge > 70 %** : le GPU devient le goulot. Réduisez le réseau (`modele.d=128` →
  1,4 M paramètres) ou augmentez `autojeu.simultanees`.
- **charge < 15 %** : le GPU a de la marge, un réseau plus gros ne coûtera presque rien.

Pour appliquer une recommandation, passez la commande complète (avec `docker compose run`, les
arguments remplacent la commande du service) :

```bash
docker compose run --rm entrainement champ entrainer --config configs/rtx-a5000-laptop.json \
    --set travailleurs=6 --set modele.d=128 --set modele.tetes=4
```

### Particularités d'un portable

- **Secteur et mode performance.** Branchez l'alimentation d'origine et choisissez le profil
  « performances » du système et du BIOS. Sur batterie, le GPU et le CPU sont fortement bridés.
- **Chaleur.** Des heures à 100 % du CPU et du GPU font chauffer. Surélevez l'arrière du
  portable et surveillez avec `nvidia-smi dmon -s pucm` (puissance, température, fréquences). Si
  la fréquence du GPU s'effondre ou si le débit (`champ suivi`, colonne parties/h) baisse au fil
  des heures, retirez un processus (`--set travailleurs=…`). La perte est souvent faible et la
  machine reste stable.
- **Interruption.** L'entraînement se reprend où il en était : on peut l'arrêter le jour
  (`Ctrl-C`) et le relancer le soir avec la même commande.
- **Pilote.** Le pilote 580 (CUDA 13.0) convient aux roues PyTorch de PyPI utilisées par l'image
  (`torch 2.14+cu130`). Si PyTorch se plaint d'un pilote trop ancien après une mise à jour,
  reconstruisez avec
  `docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 entrainement`.
- **Windows.** Utilisez WSL 2 avec Docker Desktop et le pilote NVIDIA pour Windows ; le dossier
  `runs/` doit rester dans le système de fichiers Linux (par exemple `~/champ`) : c'est beaucoup
  plus rapide que sous `/mnt/c`.
- **Option avancée sous Linux : NVIDIA MPS.** Le service multi-processus de NVIDIA laisse les
  noyaux des différents processus s'exécuter en même temps sur le GPU au lieu de se relayer. Cela
  peut réduire la charge du GPU quand elle est élevée. Pour l'essayer (entraînement lancé hors
  Docker) : `nvidia-cuda-mps-control -d`, puis `champ banc` pour comparer, et `echo quit |
  nvidia-cuda-mps-control` pour l'arrêter.

## Référence des réglages

Valeurs par défaut du code ; `configs/gpu.json` les ajuste pour un GPU. Toutes se modifient avec
`--set clé=valeur` (clés pointées pour les sous-sections, listes séparées par des virgules).

| Réglage | Défaut | Rôle |
|---|---|---|
| `dossier` | runs/principal | dossier de l'entraînement (reprise automatique) |
| `iterations` | 300 | nombre total d'itérations (détermine aussi la décroissance du taux d'apprentissage) |
| `travailleurs` | -1 | processus d'auto-jeu (-1 = cœurs physiques − 1, 0 = dans le processus principal) |
| `parties_par_iteration` | 256 | parties d'auto-jeu par itération |
| `amorce_iterations` | 2 | itérations initiales jouées avec la recherche heuristique |
| `amorce_simulations` | 32 | simulations par coup pendant l'amorçage |
| `amorce_pas` | 1 000 | pas d'apprentissage supervisé à la fin de l'amorçage |
| `amorce_melange_q` | 0,8 | part de la valeur de recherche dans la cible de valeur pendant l'amorçage |
| `autojeu.simulations` | 64 | simulations des recherches complètes (cible de politique enregistrée) |
| `autojeu.simulations_rapides` | 16 | simulations des recherches rapides (valeur seulement) |
| `autojeu.p_complete` | 0,25 | proportion de recherches complètes |
| `autojeu.max_manches` | 80 | au-delà, partie nulle (limite l'auto-jeu des réseaux faibles) |
| `autojeu.simultanees` | 64 | parties simultanées par processus (taille des lots d'inférence) |
| `autojeu.m` | 16 | actions candidates Gumbel à la racine |
| `autojeu.c_visit`, `autojeu.c_scale` | 50, 0,1 | transformation σ des Q (mctx) |
| `autojeu.meilleur_coup` | false | coup joué = meilleur coup de la recherche, sans bruit (voir « Choix du coup ») |
| `autojeu.continu` | false | moteur Rust : les parties en cours passent d'une itération à la suivante (avec le nouveau réseau) ; les lots restent pleins jusqu'au bout de l'itération, `autojeu.simultanees` n'est plus plafonné |
| `modele.d`, `modele.couches`, `modele.tetes`, `modele.ffn` | 192, 6, 6, 4 | taille du transformeur |
| `lot` | 512 | taille des lots d'apprentissage |
| `lr`, `lr_min`, `echauffement` | 1e-3, 5e-5, 300 | taux d'apprentissage (cosinus par itération) |
| `poids_decroissance` | 1e-4 | décroissance des poids (AdamW) |
| `fenetre`, `fenetre_min` | 500 000, 20 000 | taille maximale et minimale de la fenêtre d'exemples |
| `reutilisation` | 4 | nombre moyen de passages par exemple |
| `poids_valeur`, `melange_q` | 1,0, 0,25 | poids de la perte de valeur ; part de la valeur de recherche dans sa cible |
| `dispositif`, `dispositif_autojeu` | auto | `cuda`, `cpu` ou `auto` |
| `compiler` | false | `torch.compile` |
| `eval_tous`, `eval_paires`, `eval_simulations` | 5, 24, 64 | fréquence et taille des matchs d'évaluation |
| `eval_ancres` | glouton | adversaires fixes : `glouton`, `heur:64`, `mcts:800`… |
| `eval_meilleur` | true | le nouveau modèle affronte aussi le meilleur modèle (s'il n'est pas le précédent) |
| `eval_precedent` | true | le nouveau modèle affronte la version évaluée précédente |
| `eval_paires_reseaux` | 100 | paires des matchs contre un réseau (moteur Rust) ; `eval_paires` reste celui des bots |
| `eval_promotion` | elo | `elo` : meilleur.pt = meilleur Elo ; `ic` : promu seulement s'il bat le meilleur (borne basse de l'IC 95 % > 0) |
| `autojeu_reseau` | dernier | réseau de l'auto-jeu : `dernier` (le plus récent) ou `meilleur` (le meilleur démontré, `meilleur.pt`) |
| `reinit_tous`, `reinit_reutilisation` | 0, 4 | toutes les N itérations, réseau neuf entraîné depuis zéro sur la fenêtre (voir « Plasticité ») |
| `valeur_lambda` | 0 | > 0 : retours TD(λ) à la place de la valeur de recherche dans la cible de valeur |
| `inference_compilee` | false | réseau compilé (`torch.compile`, graphes CUDA) pour l'auto-jeu et les matchs Rust : +40 % de positions/s sur RTX 3070 |
| `travailleurs_evaluation` | 0 | processus des évaluations (0 = `travailleurs`) ; utile avec le moteur Rust, qui sature le GPU avec peu de processus |
| `moteur_autojeu` | auto | `auto` (Rust si le module `champ_rs` est compilé), `rust` ou `python` |
| `iterations` = 0, `lr_horizon` | —, 0 | entraînement sans fin ; décroissance du taux d'apprentissage sur `lr_horizon` itérations, puis `lr_min` |
| `modeles_gardes` | 0 | nombre maximal de modèles évalués conservés (0 = tous) |
| `releves_autojeu`, `releves_evaluation`, `releves_gardes` | 4, 2, 400 | parties enregistrées pour le spectateur (`parties/`) |
| `autojeu.duree_max` | 0 | arrêt de l'auto-jeu après N secondes (utilisé par `champ banc`) |
| `purger_donnees`, `purger_modeles` | true, true | nettoyage du disque |
| `tensorboard` | true | journal TensorBoard (si le paquet est installé) |

## Suivre l'entraînement

```bash
champ suivi runs/principal                       # tableau : pertes, nulles, débit, Elo
tensorboard --logdir runs/principal/tensorboard  # courbes (si tensorboard est installé)
```

Chaque itération ajoute une ligne à `runs/principal/journal.jsonl`. Le dossier contient :

```
runs/principal/
  config.json      configuration effective       etat.json    itération, pas, meilleur modèle
  journal.jsonl    une ligne par itération       elo.json     résultats et classement Elo
  modeles/         dernier.pt, meilleur.pt, iter_XXXX.pt (les versions évaluées)
  donnees/         exemples d'auto-jeu de la fenêtre (.npz)
  optim.pt         état de l'optimiseur          tensorboard/
```

Ce qu'il faut surveiller :

- **Elo** (le bot glouton vaut 0) : il doit croître régulièrement. Au-delà de +400, ajoutez des
  ancres plus fortes : `--set eval_ancres=glouton,heur:128,mcts:800`.
- **Taux de nulles** en auto-jeu : élevé au début (parties qui atteignent `max_manches`), il doit
  baisser rapidement après l'amorçage.
- **Parties inédites** (ligne « parties inédites » et colonne `préc*` de `champ suivi`) : avant
  d'apprendre sur les nouvelles parties d'une itération, le réseau est mesuré dessus. Ce sont
  les seules mesures de généralisation. La précision de valeur y doit dépasser nettement 0,5.
  Un grand écart avec la précision sur les exemples d'apprentissage (`préc`) signale un
  sur-apprentissage : augmentez `parties_par_iteration` ou `melange_q`, ou baissez
  `reutilisation`. Les mesures portent sur le modèle publié (poids EMA s'il y en a), celles des
  poids bruts ont le suffixe `_brut`. La **log-loss de valeur** (`logloss_valeur`) mesure
  l'étalonnage : elle doit être inférieure à celle d'une prédiction constante (`logloss_base`,
  ≈ 0,69 sans nulles) et s'approcher de celle de la valeur de la recherche (`logloss_q`). Elle
  est aussi donnée par tranche de manche (`_m1_3`, `_m4_9`, `_m10`) : en début de partie, le
  hasard domine, et un réseau trop confiant y fait pire que la constante.
- **Perte de valeur** d'apprentissage : elle baisse puis se stabilise.
- **Perte de politique** : elle baisse lentement ; elle n'est pas un bon indicateur de force à
  elle seule (les cibles deviennent plus tranchées quand le réseau progresse).

## Utiliser le bot entraîné

```bash
champ jouer --blanc humain --noir ia:400                        # 400 simulations par décision
champ jouer --blanc humain --noir ia:t=10                       # 10 secondes de réflexion par décision
champ jouer --noir "ia:modele=runs/principal/modeles/iter_0100.pt,t=2"
champ evaluer runs/principal/modeles/meilleur.pt mcts:800 --paires 50 --dispositif cuda
champ evaluer runs/principal/modeles/meilleur.pt runs/principal/modeles/iter_0050.pt
champ evaluer a.pt b.pt --sprt 0,20 --paires 100 --protocole draft   # test séquentiel (moteur Rust)
champ tournoi a.pt b.pt c.pt "c.pt:simulations=512" --paires 200 --sortie tournoi.json
champ analyser partie.nch --coup 42                             # meilleurs coups d'une position
champ analyser partie.nch --coup 42 --duree 30                  # recherche progressive de 30 s
champ analyser partie.nch --coup 42 --profondeur 12             # jusqu'à la profondeur 12
champ analyser partie.nch --coup 42 --infini                    # sans fin, une ligne par profondeur (Ctrl-C)
```

**Moteur de recherche du bot.** Si le module Rust `champ_rs` est compilé, le bot (`ia…`)
cherche avec le moteur Rust : la partie est reconstruite dans le moteur Rust à chaque coup
(`rs.etat_jeu`, `champ_rs.Jeu.depuis_etat`, testé décision par décision contre le moteur
Python), puis cherchée par vagues de 16 simulations (perte virtuelle). Au temps (`ia:t=2`) et
dans l'analyse (page Jouer, `champ analyser`), la recherche est progressive comme en Python :
passes de halving séquentiel de budget doublé sur les meilleurs candidats, l'arbre étant
conservé (`RechercheJeu.prolonger`), interruptible (analyse infinie) ; les lignes de jeu sont
les suites les plus visitées de l'arbre Rust. L'horizon affiché est alors la longueur de la plus
longue ligne principale. Le solveur exact est toujours consulté d'abord. `ia:moteur=python`
garde la recherche Python.

Le modèle par défaut est cherché dans `$CHAMP_MODELE`, `modeles/meilleur.pt`, puis
`runs/principal/modeles/meilleur.pt`. Dans l'interface web, le « Bot IA » et le bouton
« Conseil IA » apparaissent dès qu'un modèle est trouvé ; en Docker :
copiez `meilleur.pt` dans `./modeles/` puis `docker compose --profile ia up web-ia`.

## Comment ça marche

### Observation (`encodage.py`)

- **Point de vue du joueur au trait.** Tout est exprimé en « moi / adversaire ». Le plateau 2J est
  symétrique par rotation de 180° (vérifié par les tests) : Noir voit le plateau tourné, donc
  toujours sa propre armée en bas. Le réseau n'apprend qu'une seule façon de jouer.
- **46 jetons** : 37 cases (unité, camp, hauteur de pile, contrôle du Lieu, décision en attente),
  8 jetons d'unité (mes 4 unités et les 4 adverses : pièces en main, sac, défausses, réserve,
  pièces perdues, sur le plateau) et 1 jeton global (marqueurs, initiative, tailles des zones,
  Sceau royal, type de décision en attente, manche).
- **Aucune fuite d'information.** Pour l'adversaire, seul le multiensemble {sac + main +
  défausse cachée} est encodé : il se déduit de l'information publique. Un test vérifie que
  l'observation est identique quelle que soit la main réelle de l'adversaire.

### Réseau (`modele.py`)

Transformeur pré-normalisé sur les 46 jetons (attention sur tout le plateau), puis :

- **tête valeur** : probabilités Victoire / Nulle / Défaite ;
- **tête politique « pointeur »** : chaque action légale est décrite par 7 entiers (type
  d'action, pièce jouée, unité, unité ciblée ou recrutée, jusqu'à 3 cases). Son logit est calculé
  à partir des représentations des cases concernées et de plongements partagés avec l'état.
  Pas de vecteur de sortie géant : le réseau généralise aux 1 820 compositions d'armées et à
  toutes les décisions intercalées (Berserk, Moine soldat, Garde royale…).

Tailles : `cpu-demo` 0,2 M paramètres, défaut 3 M, `gpu.json` 6,5 M (d = 256, 8 couches).

### Recherche (`recherche.py`) : Gumbel IS-MCTS

- **Racine : Gumbel AlphaZero** (Danihelka et al., 2022). On tire *m* actions candidates sans
  remise (astuce Gumbel-Top-k), puis on répartit le budget par *halving séquentiel*. La cible
  d'apprentissage est la politique améliorée π' = softmax(logits + σ(Q complété)), où les Q
  complétés sont ramenés dans [0, 1] par min-max et σ(q) = (c_visit + max N) · c_scale · q, avec
  les valeurs de référence de la bibliothèque mctx de DeepMind (c_visit = 50, c_scale = 0,1).
  Sans cette normalisation, les cibles restent presque uniformes (vérifié : entropie 2,14 contre
  1,31 avec, pour 2,64 en uniforme). Cette
  méthode améliore la politique même avec 16 à 100 simulations, alors qu'AlphaZero classique en
  demande des centaines.
- **Information cachée : IS-MCTS.** Chaque simulation tire une *déterminisation* (main, sac et
  défausse cachée de l'adversaire, ordre des pioches) compatible avec ce que sait le joueur,
  puis descend dans un arbre commun indexé par les actions. Sous la racine, sélection PUCT avec
  les a priori du réseau renormalisés sur les actions disponibles, et réduction « first play
  urgency ». Les arêtes sont les actions telles que le joueur les voit : une action face cachée
  de l'adversaire (passer, recruter, prendre l'Initiative) mène au même nœud quelle que soit la
  pièce défaussée, que le joueur ignore (`recherche.cle`, testé en Python et en Rust).
- **Lots.** La recherche est un générateur qui émet des requêtes d'évaluation ; un pilote regroupe
  celles de dizaines de parties en un seul appel GPU. Le bot de jeu lance ses simulations par
  vagues de 8 grâce à la perte virtuelle.

### Choix du coup, score, recherche progressive

- **Coup joué** (`recherche.choisir`) : parmi les coups les plus explorés (au moins la moitié des
  simulations du plus exploré), celui dont le score logits + σ(Q) est le meilleur, sans bruit. Le
  bot de jeu consulte d'abord le solveur exact (`score.Solveur`, avec sa seule information) et joue
  d'office une victoire forcée. L'analyse classe les coups avec la même règle
  (`recherche.classement`) : son premier coup est celui que l'IA jouerait.
- **Score de la position** : celui du coup choisi. La moyenne de toutes les simulations de la
  racine, utilisée auparavant, compte aussi les coups médiocres explorés puis écartés par le
  halving séquentiel ; mesurée sur une partie IA contre IA, elle sous-estimait la position du
  joueur au trait de 6 points en moyenne (13 points pour une position sur dix).
- **Recherche progressive** (`RechercheGumbel.approfondir`, analyse et jeu au temps) :
  approfondissement itératif. La passe d (profondeur d) est un halving séquentiel de 32·2^(d−1)
  simulations sur les meilleurs candidats du moment ; l'arbre est conservé d'une passe à
  l'autre. Dès 1 024 simulations par passe (profondeur 6), tous les coups légaux sont candidats. Elle
  s'arrête à une durée, une profondeur, un nombre de simulations ou sur demande (analyse
  infinie). Au-delà de `max_noeuds` nœuds (≈ 3 Ko chacun), l'arbre est élagué : les nœuds peu
  visités sont retirés, les statistiques de leurs parents conservées. Chaque simulation mesure
  son horizon, en coups (pièces jouées) anticipés.
- **`meilleur_coup` en auto-jeu** : le coup joué est toujours le meilleur coup de la recherche. Le
  bruit de Gumbel ne sert plus qu'à choisir les candidats des recherches complètes, donc la
  diversité des cibles de politique. Les recherches rapides n'ont pas de bruit, comme chez
  KataGo, et un coup qui gagne immédiatement est toujours joué. La valeur de recherche mêlée à la
  cible de valeur (`melange_q`) est celle du coup joué, cohérente avec le résultat de la partie,
  joué avec les meilleurs coups. La diversité des parties vient du hasard des pioches, des
  armées et des cartes du draft.

### Auto-jeu (`autojeu.py`)

- Unités tirées au hasard à chaque partie (les 1 820 compositions possibles).
- Bruit de Gumbel à la racine pour l'exploration.
- **Playout cap randomization** (KataGo) : 25 % des coups reçoivent une recherche complète et
  produisent un exemple (cibles de politique et de valeur) ; les autres sont joués vite, sans
  exemple. On joue ainsi environ 3 fois plus de parties pour le même budget.
- **Amorçage** : les `amorce_iterations` premières itérations utilisent la recherche guidée par
  l'heuristique existante. À la fin de l'amorçage, `amorce_pas` pas d'apprentissage supervisé
  supplémentaires font imiter cette recherche au réseau. Il démarre ainsi près de son niveau, au
  lieu de partir de parties aléatoires presque toutes nulles.

### Apprentissage (`entrainement.py`)

- Perte = entropie croisée de politique vers π' + entropie croisée V/N/D vers le résultat final,
  mélangé à 25 % avec la valeur de la recherche (`melange_q`, réduit le bruit des cibles).
  Pendant l'amorçage, ce mélange monte à 80 % (`amorce_melange_q`). Avec peu de parties, le
  réseau apprend sinon par cœur l'issue de chaque partie (précision de 0,92 sur les données
  d'entraînement, environ 0,5 sur des parties jamais vues), alors que la valeur de recherche
  est une fonction de la position qui se généralise.
- AdamW, échauffement puis décroissance cosinus, écrêtage des gradients ; précision mixte bf16
  sur GPU récents (Ampere et suivants), fp16 avec mise à l'échelle des gradients sinon.
- Fenêtre glissante (`fenetre` exemples) ; chaque exemple est vu environ `reutilisation` fois.
- `compiler: true` active `torch.compile` (gain de vitesse sur GPU récent).
- **Retours TD(λ)** (`valeur_lambda` > 0, `ia/cibles.py`) : la valeur de recherche mêlée à la
  cible de valeur est remplacée par le retour λ de la position, moyenne à poids géométriques des
  valeurs de recherche des positions suivantes de la même partie (ramenées à son point de vue),
  puis du résultat final, comme la valeur « à court terme » de KataGo. λ = 0 : valeur de
  recherche seule ; λ = 1 : résultat final seul. Les exemples d'une partie sont contigus et dans
  l'ordre ; la colonne `debut` des données marque leur début (déduit de la manche pour les
  anciens fichiers, qui ne décroît jamais au cours d'une partie).

### Plasticité : réinitialisation périodique

Un réseau entraîné en continu pendant des centaines d'itérations perd sa capacité à bien
apprendre de nouvelles données (« primacy bias », Nikishin et al. 2022 ; « loss of plasticity »,
Dohare et al. 2024). Mesuré sur `runs/continu` : sur les parties inédites de l'itération 172, la
log-loss de valeur du modèle 171 (39 000 pas d'apprentissage) est de 0,73, pire qu'une prédiction
constante (0,69), alors qu'un réseau neuf entraîné depuis zéro sur la même fenêtre (itérations
146 à 171, 5 500 pas) atteint 0,58, mieux que la valeur de la recherche elle-même (0,64).
`reinit_tous: N` entraîne ainsi toutes les N itérations un réseau neuf sur la fenêtre
(`reinit_reutilisation` passages, échauffement puis décroissance cosinus), qui remplace l'ancien.

### Évaluation (`evaluation.py`)

- Matchs **appariés** : chaque graine est jouée deux fois en inversant les camps, ce qui neutralise
  la chance du tirage des unités et de l'initiative.
- Toutes les `eval_tous` itérations, le nouveau réseau affronte les ancres (`eval_ancres`) et la
  version évaluée précédente.
- Classement Elo par maximum de vraisemblance (modèle de Bradley-Terry, algorithme MM) sur tous
  les résultats. `meilleur.pt` est la version au meilleur Elo (`eval_promotion: "elo"`), ou
  (`"ic"`) celle qui a battu le meilleur modèle précédent avec une borne basse de l'intervalle
  de confiance à 95 % de l'écart Elo supérieure à 0 : un modèle n'est promu que si sa
  supériorité est démontrée.
- **Statistiques par paire** : les deux parties d'une graine ne sont pas indépendantes (même
  tirage, même draft) ; l'incertitude se calcule sur les scores de paires (0, ½, 1, 1½, 2 points,
  statistique « pentanomiale » de Fishtest). Chaque match donne l'écart Elo et son intervalle de
  confiance à 95 %. Ordre de grandeur : ±50 Elo avec 100 paires, ±35 avec 200, ±17 avec 800.
- **Moteur Rust pour les matchs entre réseaux** (`champ_rs.Match`, `evaluation.match_reseaux`) :
  toutes les parties avancent ensemble, les requêtes de chaque camp forment un lot GPU par réseau
  (un seul si les deux camps ont le même). Les agents jouent comme le bot : meilleur coup, sans
  bruit, 32 candidats, coup gagnant toujours joué (le solveur exact du bot n'est pas porté).
  `eval_paires_reseaux` paires contre chaque adversaire réseau ; les ancres peuvent être des
  modèles figés (`eval_ancres: ["glouton", "ancres/iter_0171.pt"]`, hors de `modeles/` dont les
  anciens fichiers sont purgés).
- **Test séquentiel (SPRT)** : `champ evaluer a.pt b.pt --sprt 0,20` joue des tranches de
  `--paires` paires jusqu'à décider entre H0 (écart ≤ 0) et H1 (écart ≥ 20 Elo), risques 5 %
  (approximation GSPRT sur les scores de paires). C'est la façon de valider un changement de la
  recherche à réseau égal : `--agent-b cle_publique=false`, `--agent-a c_scale=0.2`…
- **Tournoi** : `champ tournoi a.pt b.pt "a.pt:simulations=512" glouton --paires 200` fait jouer
  toutes les paires sur les mêmes graines et affiche la matrice des résultats (pas seulement un
  Elo : une relation non transitive s'y voit) et le classement Bradley-Terry.

## Génération 2 : draft, cibles auxiliaires, valeur des unités

- **Draft.** `Game(units="draft")` : 8 cartes tirées, choix A1 B2 A2 B2 A1 (actions `draft`, manche
  0), puis B commence. Une part `autojeu.p_draft` des parties d'auto-jeu (0,75) commence ainsi ;
  chaque choix reçoit une recherche complète de `autojeu.simulations_draft` simulations, à travers
  la suite de la partie. Pendant l'amorçage, `p_draft = 0` (l'heuristique ne sait pas drafter).
- **Encodage v2.** Les 8 jetons d'unité sont les cartes en jeu (à moi, adverses, disponibles), sans
  position ; 4 caractéristiques globales de draft, nulles en jeu : une position issue d'un draft
  s'encode comme la même position à armées imposées (testé). Les modèles v1 sont refusés.
- **Réseau v2.** Pointeur contextuel (le logit d'une action lit le jeton de l'unité concernée) et
  têtes auxiliaires : contrôle final de chaque Lieu, marge finale de marqueurs, main adverse
  (`poids_aux`). Les exemples de draft pèsent `poids_draft` dans la perte de politique et mélangent
  `melange_q_draft` de valeur de recherche dans leur cible de valeur.
- **EMA.** `ema` > 0 : `dernier.pt` et `meilleur.pt` publient une moyenne mobile des poids ; les poids
  bruts (reprise) sont dans `modeles/brut.pt`.
- **Évaluation.** `eval_protocole: "draft"` : matchs appariés sur les mêmes 8 cartes, camps inversés ;
  les ancres sans recherche (glouton) choisissent leurs cartes au hasard. `champ evaluer --protocole
  draft`.
- **Valeur dynamique des unités** (`unites.actif`, `ia/valeurs.py`), à chaque itération, en points du
  scoreur (10 = un bastion) :
  - K est recalibré sur la fenêtre d'exemples (`echelle.json`, utilisé aussi par `/jouer`) ;
  - 256 tirages fixes de 8 cartes × 70 répartitions, évalués par le réseau publié ;
  - régression ridge : valeur propre de chaque unité (± IC 95 % par bootstrap), synergies, contres,
    valeur de commencer ;
  - draft optimal exact sur ces sondes : fréquence de premier choix optimal, regret si on ne la prend
    pas en premier, avantage du premier à choisir ;
  - préférences réelles de l'auto-jeu (π' des cartes disponibles, 1 = neutre).

  Tout est écrit dans `unites.jsonl` (historique) et `unites.json` (dernier état). Ces valeurs ne
  rétroagissent pas sur l'apprentissage : ce sont des mesures.

## Débit et ordre de grandeur

Coût mesuré d'une simulation côté Python (déterminisation, descente, encodage) : 0,12 ms sur un
cœur de serveur modeste. Deux optimisations ont réduit ce coût de 0,195 à 0,12 ms : les actions
sont des tuples nommés (hachage en C) et les copies de recherche évitent de réinitialiser un
générateur aléatoire. Sur GPU, l'évaluation par lots ajoute peu tant que le GPU n'est pas
saturé : environ 8 000 simulations par seconde et par processus. Avec 96 / 24 simulations
(42 en moyenne) et environ 95 recherches par partie, cela fait environ 2 parties par seconde et
par processus, soit de l'ordre de 40 000 à 50 000 parties par heure avec 7 processus. Ce sont
des estimations : `champ banc` donne les chiffres réels de votre machine.

À titre indicatif, les jeux de complexité comparable demandent de l'ordre de 10⁵ à 10⁶
parties d'auto-jeu pour un niveau très fort, soit de quelques heures à quelques jours sur une
seule machine. Ces chiffres sont des estimations à confirmer avec `champ suivi` (colonne
parties/heure).

## Pistes pour aller plus loin

- **Serveur d'inférence central** (un seul processus sur le GPU qui reçoit les requêtes de tous
  les processus d'auto-jeu) pour de plus gros réseaux ; avancée des parties d'auto-jeu en
  plusieurs fils (déjà fait pour les matchs, `matchs.rs`).
- **Solveur exact du draft** : minimax sur les répartitions finales des 8 cartes (au plus 70),
  chacune évaluée par le réseau à la première décision de la manche 1 sur plusieurs tirages des
  sacs, en un seul lot, en remplacement du MCTS aux décisions de draft.
- **Analyse en Rust** : approfondissement itératif et lignes de jeu dans le moteur Rust (l'état
  s'importe déjà, `Jeu.depuis_etat`), pour la page Jouer.
- **Inférence** : double tampon CPU/GPU, TensorRT.
- **Déterminisations pondérées** par la tête de croyance sur la main adverse.
- **4 joueurs** : l'encodage actuel est spécifique au plateau 2J.
