# État au 27/09/2026 et suite

## Où en est l'entraînement (27/09/2026, fin d'après-midi)
- Run `runs/continu` (règles v2), profil local `configs/continu-rtx-3070.local.json` : d160×6,
  réinitialisation tous les 25 itérations (la première a eu lieu à l'itération 29 : 11 933 pas en
  16 min), évaluation toutes les 2 itérations contre le meilleur modèle (100 paires, décide de la
  promotion, `eval_promotion_z` 1,28) et l'ancre regles_v1_0234 (40 paires, suivi).
- Première promotion d'un réseau appris à l'itération 24 (0,58 contre l'enseignant regles_v1_0234,
  IC [+22, +94] Elo) : depuis, l'auto-jeu est joué par un réseau v2.
- Le réseau réinitialisé (itération 30) : 0,56 contre iter_0024 (+40 [−7, +89]), 0,64 contre
  regles_v1_0234 ; itération 32 : 0,55 (+35 [−10, +81]) et 0,68 (Elo +918, glouton = 0), non promu
  à 1,96 écart type, il l'aurait été à 1,28 (critère en place depuis l'itération 32).
- Temps par itération (itération 32) : auto-jeu 260 s, apprentissage 28 s, unités 25 s, évaluation
  222 s toutes les 2 itérations → ≈ 14 min pour 2 itérations (≈ 22 min le matin).
- Bot (web, analyse) : draft exact, c_scale 0,2, vagues de 16 simulations (`bots/neural.py`).

## Suite proposée
1. Laisser tourner ; suivre la progression contre regles_v1_0234 et les promotions.
2. c_scale de l'auto-jeu (cible π') : 0,2 gagne en jeu ; l'essayer à l'entraînement demande une
   comparaison de runs (deux courts runs repris de la même itération, ou une ablation hors ligne).
3. Draft exact en auto-jeu (`autojeu.draft_exact`) : gain mesuré en jeu seulement.
4. Évaluation : ≈ 30 % du temps GPU (280 parties à 128 simulations toutes les 2 itérations).
   Pistes : `eval_tous` 3, ou arrêt séquentiel (SPRT) du match contre le meilleur.
5. Débit : le GPU est le goulot (auto-jeu 89 % dans le réseau). Serveur d'inférence unique
   (un seul contexte CUDA pour les 3 processus) à mesurer.
6. Réutilisation de l'arbre entre les coups pour le bot (gain à temps de réflexion égal).

## Nouveau départ : règles des cartes v2 (27/09/2026)
- Lancé le 27/09 au matin (`make continu`).
- Règles corrigées : Moine soldat une fois par tour, Fantassin recruté déployable aussitôt, Berserk
  (après une attaque ou un déplacement, attaquer ou se déplacer à nouveau ; plus de contrôle). Les
  données et réseaux antérieurs suivaient les anciennes règles.
- Spectateur : l'ancien et le nouvel entraînement sur les mêmes courbes (et l'onglet Unités), ligne
  verticale « Règles des cartes v2 » ; réglage `predecesseur` du profil ; plus de menu de choix.
- Ancien entraînement archivé tel quel : `runs/continu_regles_v1` (meilleur : iter_0234, copié en
  `runs/ancres/regles_v1_0234.pt`).
- `runs/continu` repart de l'itération 0 : réseau appris initialisé au hasard, fenêtre vide. L'auto-jeu
  est joué par le meilleur réseau démontré (`autojeu_reseau: "meilleur"`), d'abord regles_v1_0234 (il
  ne sert qu'à produire des parties, jouées avec les nouvelles règles) ; un réseau appris le remplace dès
  qu'il le bat avec un IC 95 % > 0. Réinitialisation tous les 25 itérations.
- À surveiller : la première promotion d'un réseau appris (`meilleur` dans `champ suivi`), puis la
  progression contre l'ancre regles_v1_0234.

## Réglage de la recherche (27/09/2026, après-midi ; regles_v1_0234, règles v2, draft, 128 simulations, 150 paires)
Écart de chaque variante contre la recherche par défaut (c_scale 0,1, c_puct 1,25, fpu 0,25, m 32) :

| Variante | Elo [IC 95 %] |
|---|---|
| draft exact (`draft_exact=8`) | **+35 [−1, +72]** (avec la mesure du matin : ≈ +34 [+10, +58]) |
| c_scale 0,05 | **−73 [−114, −33]** |
| c_scale 0,2 | +18 [−17, +55] |
| c_puct 0,8 / 2,0 | +22 [−14, +58] / +7 [−29, +43] |
| fpu 0 / 0,5 | +0 [−35, +35] / −1 [−37, +34] |
| m 16 | −16 [−52, +19] |

- Le bot joue désormais le draft exact (`DRAFT_EXACT = 8`, `bots/neural.py`) ; l'auto-jeu et les
  matchs d'évaluation gardent le draft par la recherche.
- c_scale, seconde série (150 paires contre 0,1) : 0,2 **+44 [+8, +82]**, 0,3 +18 [−15, +53],
  0,5 +26 [−12, +64]. Avec la première mesure de 0,2 (+18) : ≈ +30 [+4, +56] → le bot joue
  `C_SCALE = 0.2` (`bots/neural.py`, aussi pour l'analyse). L'auto-jeu garde 0,1 : c_scale rend
  aussi la cible π' plus tranchée, effet sur l'apprentissage non mesuré.

## Taille du réseau (27/09/2026) : d160 reste le meilleur
Réseaux entraînés depuis zéro sur les mêmes données v2 (itérations 1-17, 928 000 exemples,
réutilisation 4, lots de 512 comme la réinitialisation), mesurés sur l'itération 18 :

| Réseau | log-loss valeur | KL politique | 1er coup = cible | positions/s (lots de 384) |
|---|---|---|---|---|
| d128×6 (1,4 M) | 0,5572 | 0,387 | 0,764 | 26 200 |
| **d160×6 (2,2 M)** | **0,5542** | **0,353** | **0,777** | 22 500 |
| d192×6 (3,2 M) | 0,5572 | 0,362 | 0,771 | 17 900 |

À temps égal (simulations au prorata du débit, 200 paires) : d160@128 bat d192@102 de
**+43 [+8, +78]** et d128@149 de +21 [−12, +54] (Bradley-Terry : d160 0, d128 −22, d192 −41).
À simulations égales (128) : d160 ≈ d192 (−4 [−37, +30]), d160 bat d128 de +21 [−13, +55].
La valeur ne gagne rien à la taille ; d128 n'est que 16 % plus rapide (le réseau n'est pas limité
par le calcul à cette taille). → On garde d160 (pas de `reinit_modele`).

## Débit (27/09/2026)
- L'auto-jeu passe 89 % de son temps dans le réseau (35 s contre 4,2 s pour le moteur Rust sur
  1 200 pas) : le GPU est le goulot. Ses lots font toujours 384 positions (`simultanees`), qui
  étaient complétées jusqu'au palier 512 : paliers intermédiaires (384, 768, 1536, 3072) → −22 % de
  temps de réseau par lot, +25 à 38 % de pas d'auto-jeu par seconde.
- Les matchs d'évaluation de l'entraînement (200 parties, `parallele` 1) attendaient le GPU à
  chaque simulation : 5 à 7 min toutes les 2 itérations. `eval_parallele` (4) divise d'autant les
  appels au réseau. Mais le coût de fond est le volume : 100 paires à 128 simulations par décision
  ≈ 4,4 M d'évaluations, soit 30 % de deux itérations d'auto-jeu (14,7 M) ; 200 paires en doublent
  le coût. Profil d'un match : la fin (dernières 5 % de parties) ne coûte que ≈ 10 % du temps ; la
  première évaluation après un redémarrage compile les paliers (≈ 3 min).
- Apprentissage : 7 à 9 synchronisations GPU par pas (`.item()` des pertes, sélections booléennes
  des pertes auxiliaires) supprimées, moyenne mobile en un seul `_foreach_lerp_` : −17 % par pas ;
  `compiler` (torch.compile de l'apprentissage) : −30 % de plus. Le temps par pas ne dépendait
  presque pas de la taille du réseau (d128 165 ms, d192 180 ms sous contention) : surcoût, pas calcul.
- Lots plus grands que 384 : −7 % par position à 768, −14 % à 1536 seulement (GPU déjà saturé).
- Recherche par vagues à 128 simulations (v2_0014, 200 paires) : parallele 4 −28 [−59, +2],
  parallele 8 +12 [−19, +43] → pas de coût établi ; `eval_parallele` 4 dans le profil local.
- Résultat sur l'entraînement (RTX 3070) : auto-jeu 390 → 260-300 s par itération, apprentissage
  70-90 → 35-40 s ; une évaluation (400 parties à 128 simulations) ≈ 450 s, ramenée à 280 parties
  (`eval_paires_ancres` 40 : l'ancre regles_v1_0234 ne sert plus qu'au suivi).

## Mesures du 27/09/2026 (regles_v1_0234, règles v2, draft, 128 simulations, 200 paires)
- C1 (arêtes face cachée sans pièce) contre l'ancienne recherche : +4 [−30, +37] → sans effet mesurable
  (correction conservée).
- Draft exact (`draft_exact=8`) : +33 [+1, +66] contre la recherche standard, −24 [−54, +5] contre
  l'ancienne recherche → signal faible, à confirmer par SPRT avant de l'activer en auto-jeu.
- Vagues de 16 simulations (bot Rust) à 800 simulations : −9 [−44, +25] (150 paires) → sans coût
  mesurable : le bot profite pleinement de la vitesse de la recherche Rust.
- Croyance sur la main adverse : la tête `main_adverse` (regles_v1_0234) ne fait que 5 % mieux que le
  tirage hypergéométrique sur les pièces possibles (log-loss 0,598 contre 0,632) → peu d'information
  sur la main au-delà de l'information publique ; déterminisations pondérées écartées.
- Nouveau run : le réseau neuf bat glouton à 98 % dès l'itération 2 ; contre l'enseignant : 0,12 (it. 2),
  0,20 (it. 4).

# État au 26/09/2026

## Mesures clés (RTX 3070, moteur Rust, draft, 128 simulations)
- Réseau neuf entraîné depuis zéro sur la fenêtre 146-171 : **+180 Elo** contre iter_0171,
  +220 contre iter_0172 (200 paires). Log-loss de valeur inédite 0,58 contre 0,73 : perte de
  plasticité du réseau entraîné en continu. iter_0171 ≈ iter_0172 : le plateau était réel.
- Nombre de simulations (réseau neuf) : 64 → 128 : +111 Elo ; 128 → 256 : +74 à +83.
- Cibles de valeur TD(λ) / melange_q : aucun gain mesurable hors ligne (non activées).
- torch.compile : +40 % de positions/s ; la VRAM n'est pas limitante (≈ 0,5 Go par processus).

## En place
- `runs/continu` reprend sur le réseau neuf (`neuf_0172`, copie dans `runs/ancres/`) ;
  sauvegarde de l'état antérieur : `runs/continu/sauvegarde_iter0172/`.
- Profil local `configs/continu-rtx-3070.local.json` : compilé, auto-jeu continu, auto-jeu joué
  par `meilleur.pt`, réinitialisation tous les 25 itérations, évaluation Rust contre les ancres,
  promotion si IC 95 % > 0. Profils suivis : `configs/continu-8go.json`, `continu-16go.json`.
- Bot et analyse web avec la recherche Rust (progressive, lignes, interruptible).

## À faire
1. Suivre `champ suivi runs/continu` : le réseau progresse-t-il au-delà de neuf_0172 ? Sinon,
   réinitialiser plus souvent (`reinit_tous` 10-15) ou essayer une décroissance des poids plus forte.
2. Rejouer les mesures interrompues (scripts dans le scratchpad de la session, à recréer) :
   C1 (`cle_publique=false`), draft exact (`draft_exact=8`, lots maintenant découpés),
   vagues de 16 simulations à 800 simulations, avec `champ tournoi … --simultanees 512`.
3. Réseaux plus grands entraînés depuis zéro sur la fenêtre (d=192×6, d=256×8), comparés à
   simulations égales puis à temps égal.
4. `champ banc --modele runs/continu/modeles/meilleur.pt` pour ajuster `simultanees` et `travailleurs`.
5. Reconstruire l'image (`make images`) après le dernier correctif (`evaluateurs.py`, découpage
   des lots > 8192) avant d'utiliser `draft_exact` dans le service.
6. Sous Windows, appeler l'API en `127.0.0.1:8090` (`localhost` attend 21 s sur IPv6).
