Le travail est commité (`2a9e6b5`, 37 fichiers), sur `main`, sans push. L'entraînement continue de tourner en arrière-plan. J'ai arrêté la surveillance du journal pour ne plus consommer de tokens.

## Prochaines actions, par priorité

**1. Audit des 16 cartes (avant que le run n'avance trop)**
- Comparer chaque carte (`static/img/cartes/*.jpg`) et la FAQ du livret aux deux moteurs, comme on l'a fait pour la Garde royale.
- Toute règle corrigée impose un nouveau redémarrage de zéro : plus c'est tôt, moins on perd.

**2. Suivi du run (quelques minutes par jour)**
- `make continu-journal` : le taux de nulles doit rester bas et l'Elo passer au-dessus de 0 (le niveau du glouton).
- `victoires_choisit / parties_draft` doit s'approcher de 50 % : sinon, le premier à choisir est trop avantagé, ou ne l'est pas assez.
- `runs/continu/unites.jsonl` : les valeurs des unités doivent se stabiliser, avec des intervalles de confiance qui se resserrent.

**3. Vue « Unités » du tableau de bord `/`**
- Barres de valeur ± intervalle de confiance, évolution par itération, cartes de chaleur des synergies et des contres, indicateurs du draft.
- Dans `/jouer`, un badge de valeur sur chaque carte pendant le draft (conseil au joueur humain).

**4. Force et débit, sans redémarrage**
- **Solveur exact du draft** en Rust : environ 290 évaluations au lieu de 560 par draft, et des cibles plus justes. À activer vers les itérations 10 à 15.
- **Évaluations en Rust** : elles prennent environ 27 % du temps total, et le bruit de l'Elo passerait de ±50 à ±25.
- **Inférence plus rapide** (CUDA graphs, double tampon CPU/GPU) : ×2 à ×3 estimé. C'est le préalable à un réseau plus gros (192/8/6), qui lui demandera un nouveau run.
- **Promotion de `meilleur.pt` avec une marge** statistique, et un second Elo en armées aléatoires.

**5. Plus tard**
- Fenêtre d'exemples croissante façon KataGo.
- Déterminisations pondérées par la tête de croyance sur la main adverse.
- TensorRT.
- Ligue d'adversaires.
- Draft à 4 joueurs.

Le détail de ces pistes est dans [docs/IA.md](docs/IA.md), à la rubrique « Pistes pour aller plus loin ». Je les ai aussi notées dans ma mémoire de projet, pour les retrouver au prochain échange.