// Données envoyées par champ_dhonneur/server/spectateur.py

/** [nom, x, y2, lieu] */
export type Case = [string, number, number, boolean];

export interface Carte {
  nom: string;
  pieces: number;
  tactique: string;
  capacite: string;
}

export interface Decor {
  mode: string;
  cols: number;
  max_y2: number;
  cases: Case[];
  equipes: number[];
  unites: string[][];
  cartes: Record<string, Carte>;
  graine: number;
  entetes: Record<string, string>;
  /** mise en place avancée : les 8 cartes tirées, joueur qui choisit la première */
  draft?: { cartes: string[]; premier: number };
}

/** [identifiant, case, joueur, type, pièces] */
export type UniteImg = [number, number, number, string, number];

export interface JoueurImg {
  h: string[]; // main
  s: number; // pièces dans le sac
  d: string[]; // défausse visible
  x: number; // défausse cachée
  v: Record<string, number>; // réserve
  l: number; // pièces éliminées
}

export interface ActionImg {
  n: string; // notation
  d: string; // description
  j: number; // joueur
  k: string; // type d'action
  c: number[]; // cases
  u: string | null;
  pc: string | null;
  ti?: string[]; // partie hybride : pièces piochées par l'IA à la suite de ce coup
}

export interface Image {
  r: number; // manche
  t: number; // joueur au trait
  i: number; // initiative
  m: [number, number]; // marqueurs restant à poser
  u: UniteImg[];
  c: [number, number][]; // [case, équipe] des lieux contrôlés
  j: JoueurImg[];
  a: ActionImg | null; // décision qui mène à cette image
  p?: string; // décision en attente
  f?: { g: number | null; r: string }; // fin de partie
  arm?: string[][]; // armées courantes (parties avec draft)
  tir?: { dispo: string[]; etape: number }; // draft en cours : cartes encore disponibles
}

export interface Film {
  id: string;
  decor: Decor;
  images: Image[];
  fini: boolean;
}

export interface TableDirect extends Film {
  table: string;
  parties: number;
  total: number;
  maj: number;
}

export interface Phase {
  phase: "autojeu" | "apprentissage" | "evaluation" | string;
  iteration: number;
  debut: number;
  joueur?: string;
  amorce?: boolean;
}

export interface ResumePartie {
  fichier: string;
  type: string;
  iteration: number;
  blanc: string;
  noir: string;
  resultat: string;
  unites: [string, string];
  manches: number;
  date: number;
}

export interface PointElo {
  iteration: number;
  elo: number;
  meilleur?: string;
  matchs: Record<string, number>;
}

export type Series = Record<
  | "iteration" | "parties" | "parties_par_heure" | "manches" | "nulles" | "victoires_blanc"
  | "exemples" | "fenetre" | "perte_politique" | "perte_valeur" | "entropie" | "precision"
  | "val_precision" | "val_premier_coup" | "val_kl" | "t_autojeu" | "t_apprentissage"
  | "t_evaluation",
  (number | null)[]
> & { amorce: boolean[] };

export interface Tableau {
  nom: string;
  maintenant: number;
  iteration: number;
  meilleur: string | null;
  elo_meilleur: number | null;
  ancres: Record<string, number>;
  config: {
    iterations: number;
    modele: { d?: number; couches?: number; tetes?: number };
    simulations?: number;
    parties_par_iteration?: number;
    travailleurs?: number;
    eval_tous?: number;
  };
  phase: Phase | null;
  series: Series;
  courbe: PointElo[];
  parties: ResumePartie[];
  totaux: { parties: number; secondes: number };
  maj: number | null;
}

export interface Run {
  nom: string;
  iteration: number;
  meilleur: string | null;
  maj: number;
}

/** Valeur dynamique des unités (runs/<nom>/unites.jsonl, écrit par ia/valeurs.py), en points (10 = un bastion) */
export interface ValeurUnite {
  points: number;
  ic95: number;
  premier_choix: number; // fréquence de premier choix optimal quand la carte est tirée
  regret_premier: number; // points perdus si A ne la prend pas en premier
  choix_autojeu?: number; // préférence de l'auto-jeu (1 = neutre)
  premier_autojeu?: number;
}

export interface MesureUnites {
  iteration: number;
  K: number;
  unites: Record<string, ValeurUnite>;
  synergies: Record<string, number>; // « AB » : A et B alliés
  contres: Record<string, number>; // « AB » > 0 : A l'emporte sur B
  commencer: number;
  avantage_premier_choix: number;
  r2: number;
  r2_effets_propres: number;
  sondes: number;
  decisions_draft: number;
}

export interface Unites {
  historique: MesureUnites[];
  draft: { iteration: number; parties_draft: number; victoires_choisit: number }[];
}
