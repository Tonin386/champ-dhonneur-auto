# État au 26/09/2026 et suite

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
