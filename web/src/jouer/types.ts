// Données échangées avec champ_dhonneur/server/jeu.py (page « Jouer »)
import type { Case, Decor, Image } from "../types";

export interface JoueurCfg {
  type: "humain" | "ia";
  niveau: number; // simulations par décision
  modele: string | null;
  nom?: string;
}

export interface Legal {
  i: number;
  n: string; // notation publique
  d: string; // description
  coin: string | null;
  kind: string;
  cells: number[];
}

export interface EtatServeur {
  id: string;
  version: number;
  debut: number;
  n: number;
  images: Image[];
  decor?: Decor;
  joueurs: JoueurCfg[];
  trait: number;
  humain: boolean;
  ia: boolean;
  legal: Legal[];
  attente: string;
  piece_attente: string | null;
  /** draft : valeur de chaque carte disponible pour le joueur qui choisit (ia/conseil.py) */
  conseil?: Conseil | null;
  fini: boolean;
  resultat: string;
  edite: boolean;
  mise: Mise;
  mains_visibles: boolean;
  masques: number[];
  /** partie hybride (plateau réel) : sièges de l'IA et du joueur plateau */
  hybride: { ia: number; plateau: number } | null;
  /** partie hybride : pioche de l'IA à saisir */
  tirage: Tirage | null;
  /** partie hybride : pièces que le joueur plateau pourrait jouer (nombre de chaque) */
  possibles: Record<string, number> | null;
}

/** Pioche de l'IA à saisir, pièce par pièce (partie hybride) */
export interface Tirage {
  contexte: "initial" | "manche" | "moine";
  sac: Record<string, number>; // pièces du sac de l'IA (après remise de la défausse)
  melange: boolean; // sac vide : la défausse de l'IA y est remise d'abord
  rang: number;
  total: number;
  choisies: string[];
  coup: { n: string; d: string; j: number } | null; // coup qui déclenche la pioche
}

/** Mise en place de la partie (draft, libre ou position de l'éditeur) */
export interface Mise {
  mode: "draft" | "libre" | "position";
  libelle: string;
}

export interface CoupAnalyse {
  coup: string;
  description: string;
  probabilite: number;
  q: number | null;
  visites: number;
  score: number;
  texte: string;
  ligne: string[];
  cases: number[];
  i: number; // index du coup parmi les coups légaux
}

export interface Analyse {
  pos: number;
  trait: number;
  bastions: [number, number];
  source?: "reseau" | "materiel";
  score?: number;
  texte?: string;
  appreciation?: [string, string];
  coups: CoupAnalyse[];
  mat?: { equipe: number; coups: number; coup: string | null } | null;
  indisponible?: string;
  fini?: string;
  simulations?: number;
  v_or?: number;
  gain_or?: number; // part de la barre d'évaluation revenant à Or, dans [0, 1]
}

export interface UniteRegle {
  nom: string;
  pieces: number;
  max: number;
  tactique: string;
  capacite: string;
}

export interface Regles {
  cases: Case[];
  cols: number;
  max_y2: number;
  departs: string[][];
  unites: Record<string, UniteRegle>;
  premiere: string[][];
  armees: Record<string, string[][]>; // armées toutes faites du mode libre
  draft: number; // nombre de cartes du draft
  niveaux: Record<string, string>;
}

export interface InfoIA {
  disponible: boolean;
  raison?: string;
  modeles: { chemin: string; nom: string }[];
}

/** Position de l'éditeur (format de champ_dhonneur/position.py) */
export interface Position {
  unites: string[][];
  plateau: { case: string; joueur: number; unite: string; pieces: number }[];
  controle: Record<string, number>;
  trait: number;
  initiative: number;
  manche: number;
  joueurs: {
    main: string[];
    sac: string[];
    defausse: string[];
    defausse_cachee: string[];
    reserve: Record<string, number>;
  }[];
}

export interface Conseil {
  cartes: Record<string, { valeur: number; propre: number; ic95: number; synergie: number; contre: number }>;
  meilleure: string;
  iteration: number | null;
  source: string | null;
  joueur: number;
}
