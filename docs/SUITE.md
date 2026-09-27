# État au 27/09/2026 et suite

## Nouveau départ : règles des cartes v2 (27/09/2026)
- **Pour lancer demain** : `make continu` (ou `docker compose --profile continu up -d`). Le run est prêt
  à l'itération 0 et n'a pas été lancé.
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
- c_scale : pente nette vers le haut (0,05 ≪ 0,1 ≤ 0,2) → essayer 0,3 et 0,5 (bot et cible π').

## Débit (27/09/2026)
- L'auto-jeu passe 89 % de son temps dans le réseau (35 s contre 4,2 s pour le moteur Rust sur
  1 200 pas) : le GPU est le goulot. Ses lots font toujours 384 positions (`simultanees`), qui
  étaient complétées jusqu'au palier 512 : paliers intermédiaires (384, 768, 1536, 3072) → −22 % de
  temps de réseau par lot, +25 à 38 % de pas d'auto-jeu par seconde.
- Les matchs d'évaluation de l'entraînement (200 parties, `parallele` 1) attendaient le GPU à
  chaque simulation : 5 à 7 min toutes les 2 itérations. `eval_parallele` (4) divise d'autant les
  appels au réseau.

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
