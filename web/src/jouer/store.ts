import { create } from "zustand";
import type { Decor, Image } from "../types";
import type { Analyse, Conseil, EtatServeur, InfoIA, JoueurCfg, Legal, Mise, Position, Regles, Tirage } from "./types";

export interface Config {
  joueurs: JoueurCfg[];
  mode: "draft" | "libre";
  /** draft : les cartes choisies (null : tirées au hasard) */
  cartes: string[] | null;
  /** libre : les deux armées de 4 unités */
  armees: string[][];
  /** draft : premier à choisir ; libre : Initiative (null : au hasard) */
  premier: number | null;
  graine: number | null;
  mains_visibles: boolean;
  /** partie sur un vrai plateau : une IA, un joueur plateau (docs/HYBRIDE.md) */
  hybride: boolean;
}

export const CONFIG_DEFAUT: Config = {
  joueurs: [{ type: "humain", niveau: 200, modele: null }, { type: "ia", niveau: 200, modele: null }],
  mode: "draft",
  cartes: null,
  armees: [["H", "P", "S", "X"], ["A", "C", "E", "L"]],
  premier: null,
  graine: null,
  mains_visibles: false,
  hybride: false,
};

/** Configuration enregistrée (les anciennes clés, comme « unites », sont oubliées). */
function lireConfig(): Config {
  const c = lire("champ.jouer.config", CONFIG_DEFAUT) as Config & { unites?: string };
  const { unites: _, ...reste } = c;
  if (!["draft", "libre"].includes(reste.mode)) reste.mode = "draft";
  return reste;
}

export type Mode = "jeu" | "editeur";

interface EtatJeu {
  regles: Regles | null;
  infoIA: InfoIA | null;
  config: Config;

  // partie (miroir du serveur)
  id: string | null;
  version: number;
  decor: Decor | null;
  images: Image[];
  joueurs: JoueurCfg[];
  trait: number;
  humain: boolean;
  ia: boolean;
  legal: Legal[];
  attente: string;
  pieceAttente: string | null;
  conseil: Conseil | null;
  fini: boolean;
  resultat: string;
  edite: boolean;
  mise: Mise | null;
  mainsVisibles: boolean;
  masques: number[];
  hybride: { ia: number; plateau: number } | null;
  tirage: Tirage | null;
  possibles: Record<string, number> | null;

  // affichage
  pos: number; // image affichée
  occupe: boolean; // requête de jeu en cours
  lecture: boolean; // IA contre IA : lecture automatique
  vitesse: number; // ms entre deux décisions de l'IA
  piece: string | null; // pièce de la main choisie
  caseChoisie: number | null;
  survol: number[] | null; // cases du coup survolé dans une liste
  mode: Mode;
  dialogue: boolean;
  onglet: "coups" | "analyse";

  // analyse
  analyse: boolean;
  simulations: number;
  fleche: boolean;
  analyses: Record<number, Analyse>;
  enAnalyse: number | null;

  editeur: Position | null;
  erreur: string | null;
  info: string | null;

  demarrer(): Promise<void>;
  nouvelle(c?: Partial<Config>, position?: Position): Promise<boolean>;
  jouer(i: number): Promise<void>;
  coupIA(): Promise<void>;
  revenir(pos: number): Promise<void>;
  annuler(): Promise<void>;
  reglages(r: { joueurs?: JoueurCfg[]; mains_visibles?: boolean }): Promise<void>;
  piocher(piece: string): Promise<void>;
  recommencerPioche(): Promise<void>;
  aller(pos: number): void;
  pas(d: number): void;
  choisirPiece(c: string | null): void;
  choisirCase(i: number | null): void;
  setSurvol(c: number[] | null): void;
  analyser(pos: number): Promise<void>;
  basculerAnalyse(): void;
  setSimulations(n: number): void;
  ouvrirEditeur(): Promise<void>;
  setEditeur(p: Position | null): void;
  setErreur(e: string | null): void;
  set(p: Partial<EtatJeu>): void;
}

async function api<T>(chemin: string, corps?: unknown): Promise<T> {
  const r = await fetch(chemin, corps === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corps),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : r.statusText);
  return j as T;
}

function lire<T>(cle: string, defaut: T): T {
  try {
    const v = localStorage.getItem(cle);
    return v === null ? defaut : { ...defaut, ...JSON.parse(v) };
  } catch {
    return defaut;
  }
}

function ecrire(cle: string, v: unknown) {
  try {
    localStorage.setItem(cle, JSON.stringify(v));
  } catch {
    /* stockage indisponible */
  }
}

const prefs = lire("champ.jouer", { analyse: false, simulations: 400, fleche: true, vitesse: 700 });

export const useJeu = create<EtatJeu>()((set, get) => {
  const erreur = (e: unknown) => set({ erreur: (e as Error).message, occupe: false });
  const q = () => `depuis=${get().images.length}&version=${get().version}`;

  /** Nouvelles images : prolonge ou remplace ; l'affichage suit le direct s'il y était. */
  const recevoir = (e: EtatServeur) => {
    const s = get();
    const auBout = s.pos >= s.images.length - 1;
    const images = e.debut === 0 ? e.images : s.images.slice(0, e.debut).concat(e.images);
    const nouvelle = e.id !== s.id;
    // positions effacées (retour arrière) ou information différente : analyses à refaire
    let analyses = s.analyses;
    if (nouvelle || e.masques.join() !== s.masques.join()) analyses = {};
    else if (e.debut < s.images.length) {
      analyses = Object.fromEntries(Object.entries(analyses).filter(([k]) => +k < e.debut));
    }
    set({
      id: e.id, version: e.version, decor: e.decor ?? s.decor, images,
      joueurs: e.joueurs, trait: e.trait, humain: e.humain, ia: e.ia, legal: e.legal,
      attente: e.attente, pieceAttente: e.piece_attente, conseil: e.conseil ?? null, fini: e.fini, resultat: e.resultat,
      edite: e.edite, mise: e.mise, mainsVisibles: e.mains_visibles, masques: e.masques, analyses,
      hybride: e.hybride ?? null, tirage: e.tirage ?? null, possibles: e.possibles ?? null,
      ...(e.tirage && !s.tirage ? { onglet: "coups" as const } : {}),
      pos: auBout || nouvelle || s.pos >= images.length ? images.length - 1 : s.pos,
      piece: null, caseChoisie: null, survol: null, occupe: false,
    });
  };

  return {
    regles: null,
    infoIA: null,
    config: lireConfig(),
    id: null,
    version: 0,
    decor: null,
    images: [],
    joueurs: [],
    trait: 0,
    humain: false,
    ia: false,
    legal: [],
    attente: "",
    pieceAttente: null,
    conseil: null,
    fini: false,
    resultat: "*",
    edite: false,
    mise: null,
    mainsVisibles: false,
    masques: [],
    hybride: null,
    tirage: null,
    possibles: null,
    pos: 0,
    occupe: false,
    lecture: true,
    vitesse: prefs.vitesse,
    piece: null,
    caseChoisie: null,
    survol: null,
    mode: "jeu",
    dialogue: false,
    onglet: "coups",
    analyse: prefs.analyse,
    simulations: prefs.simulations,
    fleche: prefs.fleche,
    analyses: {},
    enAnalyse: null,
    editeur: null,
    erreur: null,
    info: null,

    set: p => set(p),
    setErreur: erreur => set({ erreur }),

    demarrer: async () => {
      try {
        const [regles, infoIA] = await Promise.all([api<Regles>("/api/jeu/regles"), api<InfoIA>("/api/jeu/ia")]);
        set({ regles, infoIA });
        const c = get().config;
        // sans réseau disponible : partie entre humains
        if (!infoIA.disponible && (c.hybride || c.joueurs.some(j => j.type === "ia"))) {
          set({ config: { ...c, hybride: false, joueurs: c.joueurs.map(j => ({ ...j, type: "humain" as const })) } });
        }
        await get().nouvelle();
      } catch (e) {
        erreur(e);
      }
    },

    nouvelle: async (c, position) => {
      const config = { ...get().config, ...c };
      try {
        const e = await api<EtatServeur>("/api/jeu", { ...config, position: position ?? null });
        set({ config, mode: "jeu", editeur: null, dialogue: false, lecture: true });
        ecrire("champ.jouer.config", config);
        recevoir(e);
        return true;
      } catch (err) {
        erreur(err);
        return false;
      }
    },

    jouer: async i => {
      const { id, occupe } = get();
      if (!id || occupe) return;
      set({ occupe: true });
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/jouer?${q()}`, { index: i }));
      } catch (e) {
        erreur(e);
      }
    },

    coupIA: async () => {
      const { id, occupe } = get();
      if (!id || occupe) return;
      set({ occupe: true });
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/ia?${q()}`, {}));
      } catch (e) {
        erreur(e);
        set({ lecture: false });
      }
    },

    revenir: async pos => {
      const { id } = get();
      if (!id) return;
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/revenir`, { pos }));
        set({ pos: get().images.length - 1, lecture: false });
      } catch (e) {
        erreur(e);
      }
    },

    annuler: async () => {
      const { id } = get();
      if (!id) return;
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/annuler`, {}));
        set({ pos: get().images.length - 1, lecture: false });
      } catch (e) {
        erreur(e);
      }
    },

    reglages: async r => {
      const { id } = get();
      if (!id) return;
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/reglages`, r));
        if (r.joueurs || r.mains_visibles !== undefined) {
          const config = { ...get().config, ...r };
          set({ config });
          ecrire("champ.jouer.config", config);
        }
      } catch (e) {
        erreur(e);
      }
    },

    piocher: async piece => {
      const { id, occupe } = get();
      if (!id || occupe) return;
      set({ occupe: true });
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/tirage?${q()}`, { piece }));
      } catch (e) {
        erreur(e);
      }
    },

    recommencerPioche: async () => {
      const { id } = get();
      if (!id) return;
      try {
        recevoir(await api<EtatServeur>(`/api/jeu/${id}/tirage/annuler?${q()}`, {}));
      } catch (e) {
        erreur(e);
      }
    },

    aller: pos => {
      const n = get().images.length;
      set({ pos: Math.max(0, Math.min(pos, n - 1)), piece: null, caseChoisie: null, survol: null });
    },
    pas: d => {
      get().aller(get().pos + d);
      if (get().joueurs.every(j => j.type === "ia")) set({ lecture: false });
    },

    choisirPiece: c => set({ piece: get().piece === c ? null : c, caseChoisie: null }),
    choisirCase: i => set({ caseChoisie: get().caseChoisie === i ? null : i }),
    setSurvol: survol => set({ survol }),

    analyser: async pos => {
      const { id, simulations, analyses } = get();
      if (!id || analyses[pos]) return;
      set({ enAnalyse: pos });
      try {
        const a = await api<Analyse>(`/api/jeu/${id}/analyse?pos=${pos}&simulations=${simulations}`, {});
        // une réponse arrivée après un changement de partie est ignorée
        if (get().id === id) set({ analyses: { ...get().analyses, [pos]: a } });
      } catch (e) {
        erreur(e);
      } finally {
        if (get().enAnalyse === pos) set({ enAnalyse: null });
      }
    },

    basculerAnalyse: () => {
      const analyse = !get().analyse;
      set({ analyse, onglet: analyse ? "analyse" : "coups" });
      ecrire("champ.jouer", { analyse, simulations: get().simulations, fleche: get().fleche, vitesse: get().vitesse });
    },
    setSimulations: simulations => {
      set({ simulations, analyses: {} });
      ecrire("champ.jouer", { analyse: get().analyse, simulations, fleche: get().fleche, vitesse: get().vitesse });
    },

    ouvrirEditeur: async () => {
      const { id, pos } = get();
      try {
        const p = id ? await api<Position & { hasard?: string }>(`/api/jeu/${id}/position?pos=${pos}`) : null;
        set({ editeur: p, mode: "editeur", lecture: false });
        if (p?.hasard) set({ info: p.hasard });
      } catch (e) {
        erreur(e);
      }
    },
    setEditeur: editeur => set({ editeur }),
  };
});

export const sauverPrefs = () => {
  const s = useJeu.getState();
  ecrire("champ.jouer", { analyse: s.analyse, simulations: s.simulations, fleche: s.fleche, vitesse: s.vitesse });
};
