import { create } from "zustand";
import type { Film, Run, Tableau, TableDirect } from "./types";

export type Source = { type: "direct"; table: string } | { type: "partie"; fichier: string };
export type Vue = "plateau" | "mosaique";

/** Une table est « active » si sa partie avance encore (sinon : fin d'auto-jeu, entraînement arrêté). */
export const ACTIVITE_S = 45;

interface Etat {
  runs: Run[];
  run: string | null;
  tableau: Tableau | null;
  connecte: boolean;
  decalage: number; // horloge du serveur - horloge locale, en secondes
  tables: Record<string, TableDirect>;
  source: Source | null;
  film: Film | null; // partie affichée sur le grand plateau
  pos: number; // image affichée
  lecture: boolean;
  vitesse: number; // millisecondes par décision
  vue: Vue;
  regie: boolean; // régie automatique : enchaîne direct et rediffusions
  vues: Set<string>; // rediffusions déjà montrées par la régie
  erreur: string | null;

  setRuns(r: Run[]): void;
  choisirRun(nom: string): void;
  recevoirTableau(t: Tableau): void;
  recevoirTable(msg: TableMsg): void;
  setConnecte(c: boolean): void;
  regarderTable(nom: string): void;
  regarderPartie(fichier: string): Promise<void>;
  aller(pos: number): void;
  pas(d: number): void;
  basculerLecture(): void;
  allerDirect(): void;
  setVitesse(v: number): void;
  setVue(v: Vue): void;
  basculerRegie(): void;
  demarrer(): void;
  suivante(): void;
  setErreur(e: string | null): void;
}

export interface TableMsg {
  table: string;
  retire?: boolean;
  id?: string;
  debut?: number;
  images?: Film["images"];
  decor?: Film["decor"];
  fini?: boolean;
  parties?: number;
  total?: number;
  maj?: number;
}

export const maintenant = (decalage: number) => Date.now() / 1000 + decalage;

export function tableActive(t: TableDirect, decalage: number): boolean {
  return !t.fini && maintenant(decalage) - t.maj < ACTIVITE_S;
}

/** Table à suivre de préférence : active, la partie la plus avancée. */
function meilleureTable(tables: Record<string, TableDirect>, decalage: number, sauf?: string): string | null {
  const actives = Object.values(tables).filter(t => tableActive(t, decalage) && t.table !== sauf);
  if (!actives.length) return null;
  actives.sort((a, b) => b.images.length - a.images.length);
  return actives[0].table;
}

async function charger(run: string, fichier: string): Promise<Film> {
  const r = await fetch(`/api/entrainements/${encodeURIComponent(run)}/parties/${encodeURIComponent(fichier)}`);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j as Film;
}

const lireVitesse = () => {
  try {
    return Number(localStorage.getItem("champ.vitesse")) || 800;
  } catch {
    return 800;
  }
};

export const useStore = create<Etat>()((set, get) => ({
  runs: [],
  run: null,
  tableau: null,
  connecte: false,
  decalage: 0,
  tables: {},
  source: null,
  film: null,
  pos: 0,
  lecture: true,
  vitesse: lireVitesse(),
  vue: "plateau",
  regie: true,
  vues: new Set(),
  erreur: null,

  setRuns: runs => set({ runs }),
  choisirRun: nom =>
    set({ run: nom, tableau: null, tables: {}, source: null, film: null, pos: 0, vues: new Set() }),
  setConnecte: connecte => set({ connecte }),
  setErreur: erreur => set({ erreur }),

  recevoirTableau: t => set({ tableau: t, decalage: t.maintenant - Date.now() / 1000 }),

  recevoirTable: msg => {
    const { tables, source, film } = get();
    if (msg.retire) {
      const { [msg.table]: _, ...reste } = tables;
      set({ tables: reste });
      return;
    }
    const ancienne = tables[msg.table];
    const images =
      ancienne && ancienne.id === msg.id && msg.debut
        ? ancienne.images.slice(0, msg.debut).concat(msg.images ?? [])
        : (msg.images ?? []);
    const t: TableDirect = {
      table: msg.table,
      id: msg.id!,
      decor: msg.decor ?? ancienne!.decor,
      images,
      fini: !!msg.fini,
      parties: msg.parties ?? 0,
      total: msg.total ?? 0,
      maj: msg.maj ?? 0,
    };
    const maj: Partial<Etat> = { tables: { ...tables, [msg.table]: t } };
    // la partie regardée continue : le grand plateau suit (la suivante attendra la fin de celle-ci)
    if (source?.type === "direct" && source.table === msg.table && film?.id === t.id) maj.film = t;
    set(maj);
  },

  regarderTable: nom => {
    const t = get().tables[nom];
    if (!t) return;
    set({ source: { type: "direct", table: nom }, film: t, pos: t.images.length - 1, lecture: true, vue: "plateau" });
  },

  regarderPartie: async fichier => {
    const run = get().run;
    if (!run) return;
    try {
      const f = await charger(run, fichier);
      const vues = new Set(get().vues).add(fichier);
      set({ source: { type: "partie", fichier }, film: f, pos: 0, lecture: true, vues, vue: "plateau" });
    } catch (e) {
      set({ erreur: (e as Error).message });
    }
  },

  aller: pos => {
    const f = get().film;
    if (!f) return;
    set({ pos: Math.max(0, Math.min(pos, f.images.length - 1)) });
  },
  pas: d => {
    get().aller(get().pos + d);
    set({ lecture: false });
  },
  basculerLecture: () => {
    const { lecture, film, pos } = get();
    // relancer une partie terminée la reprend du début
    if (!lecture && film && film.fini && pos >= film.images.length - 1) set({ pos: 0 });
    set({ lecture: !lecture });
  },
  allerDirect: () => {
    const { source, tables, decalage } = get();
    const nom = source?.type === "direct" && tables[source.table] ? source.table : meilleureTable(tables, decalage);
    if (nom) get().regarderTable(nom);
  },
  setVitesse: vitesse => {
    try {
      localStorage.setItem("champ.vitesse", String(vitesse));
    } catch {
      /* stockage indisponible */
    }
    set({ vitesse });
  },
  setVue: vue => set({ vue }),
  basculerRegie: () => set({ regie: !get().regie }),

  /** Premier choix à l'ouverture : une partie en direct, sinon la dernière rediffusion. */
  demarrer: () => {
    const { tables, decalage, tableau } = get();
    const nom = meilleureTable(tables, decalage);
    if (nom) get().regarderTable(nom);
    else if (tableau?.parties.length) get().regarderPartie(tableau.parties[0].fichier);
  },

  /** Fin de la partie affichée : la régie choisit la suite. */
  suivante: () => {
    const { source, tables, decalage, regie, tableau, vues } = get();
    if (source?.type === "direct") {
      const t = tables[source.table];
      // même table, partie suivante : rejouée depuis le début, elle rattrape vite le direct
      if (t && t.id !== get().film?.id && (tableActive(t, decalage) || !regie)) {
        set({ film: t, pos: 0, lecture: true });
        return;
      }
      if (!regie) return set({ lecture: false });
      const autre = meilleureTable(tables, decalage, source.table);
      if (autre) return get().regarderTable(autre);
    } else if (!regie) {
      return set({ lecture: false });
    }
    const nom = meilleureTable(tables, decalage);
    if (nom) return get().regarderTable(nom);
    const parties = tableau?.parties ?? [];
    const suite = parties.find(p => !vues.has(p.fichier)) ?? parties[0];
    if (suite) get().regarderPartie(suite.fichier);
    else set({ lecture: false });
  },
}));
